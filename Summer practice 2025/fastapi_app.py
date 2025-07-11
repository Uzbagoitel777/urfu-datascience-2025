from datetime import datetime
from decimal import Decimal
from math import pow
from sqlite3 import connect, Cursor

from fastapi import FastAPI
from pandas import DataFrame
from pydantic import BaseModel, field_validator
from typing import Optional
from requests import get

import models2 as ml
from Database_misc import db_currency_init, db_fetch_conversion


def db_connect():
    conn = connect('basodate.db')
    cursor = Cursor(conn)
    return conn, cursor

conn, cursor = db_connect()

SECRET_KEY = 'babijon228'
MoneyDecimal = Decimal("0.00")

app = FastAPI(title="Kickstarter campaign calculator", version="1.0.0")

training_results = []


class ModelSelection(BaseModel):
    Logistic: bool
    Tree: bool
    MLP: bool

    def list_all(self):
        models = []
        if self.Logistic:
            models.append('Logistic')
        if self.Tree:
            models.append('Tree')
        if self.MLP:
            models.append('MLP')
        return models

class CrowdfundingCampaign(BaseModel):
    name: str
    category: str
    main_category: str
    currency: str
    deadline: datetime
    goal: Decimal
    launched: datetime
    pledged: Decimal
    state: str
    backers: int
    country: str
    usd_pledged: Decimal
    usd_pledged_real: Decimal
    usd_goal_real: Decimal

    def normalize_money(self):
        if self.goal <= 0 or self.pledged <= 0:
            raise ValueError('Amount must be greater than 0')
        else:
            self.usd_pledged = Decimal(convert_currency(self.pledged, self.currency)).quantize(MoneyDecimal)
            self.usd_pledged_real = self.usd_pledged
            self.usd_goal_real = Decimal(convert_currency(self.goal, self.currency)).quantize(MoneyDecimal)
        return self

    def convert_to_df(self):
        dict_ = {
            'name': self.name, 'category': self.category, 'main_category': self.main_category,
            'currency': self.currency, 'deadline': self.deadline, 'goal': self.goal, 'launched': self.launched,
            'pledged': self.pledged, 'state': self.state, 'backers': self.backers, 'country': self.country,
            'usd_pledged': self.usd_pledged, 'usd_pledged_real': self.usd_pledged_real,
            'usd_goal_real': self.usd_goal_real
        }
        return DataFrame([dict_])


class AnalysisRequest(BaseModel):
    models: ModelSelection
    parameters: CrowdfundingCampaign


class AnalysisResult(BaseModel):
    Logistic_Success: Optional[bool]
    Logistic_Prob: Optional[float]
    Tree_Success: Optional[bool]
    Tree_Prob: Optional[float]
    MLP_Success: Optional[bool]
    MLP_Prob: Optional[float]


def convert_currency(sum, currency, target_currency = 'USD'):
    if currency == 'USD':
        return sum
    return sum * Decimal(get_conversion_rate(currency))


#
def fetch_conversion_rates():
    response = get('https://api.currencylayer.com/live?access_key=67c33ea781559bd6d1ab8977ada0ebda&format=1')
    data = response.json()['quotes']

    return data


def get_conversion_rate(currency):
    try:
        data = fetch_conversion_rates()
    except Exception as e:
        print(e)
        data = db_fetch_conversion(cursor, currency)
    query_string = 'USD' + currency.upper().strip()

    return pow(data[query_string], -1)


@app.on_event("startup")
async def startup_event():
    analysis_results = ml.main_analysis(ml.df, pretrained=True)
    for field in analysis_results:
        training_results.append(field)
    models = list(analysis_results).pop(0)

    return models

@app.post("/analyze", response_model = AnalysisResult)
async def analyze(request: AnalysisRequest):
    response = {
        'Logistic': None,
        'Tree': None,
        'MLP': None
    }
    models, scaler, feature_columns = training_results[:3]
    params = request.parameters
    if params.goal <= 0:
        raise ValueError('Amount must be greater than 0')
    else:
        params.usd_pledged = Decimal(convert_currency(params.pledged, params.currency)).quantize(MoneyDecimal)
        params.usd_pledged_real = params.usd_pledged
        params.usd_goal_real = Decimal(convert_currency(params.goal, params.currency)).quantize(MoneyDecimal)
    for model in request.models.list_all():
        if model:
            params_df_normalized = ml.normalize_columns(params.convert_to_df())[0]
            response[model] = ml.predict_campaign_success(models, scaler, feature_columns, params_df_normalized, model)

    log_s, log_p, tree_s, tree_p, mlp_s, mlp_p = [None]*6
    if response['Logistic']:
        log_s, log_p = response['Logistic']
        log_s = bool(log_s)
    if response['Tree']:
        tree_s, tree_p = response['Tree']
        tree_s = bool(tree_s)
    if response['MLP']:
        mlp_s, mlp_p = response['MLP']
        mlp_s = bool(mlp_s)
    return AnalysisResult(
        Logistic_Success = log_s,
        Logistic_Prob = log_p,
        Tree_Success = tree_s,
        Tree_Prob = tree_p,
        MLP_Success = mlp_s,
        MLP_Prob = mlp_p
    )


@app.get("/")
async def root():
    return {'status': 'Alive'}


@app.get(f"/{SECRET_KEY}")
async def weird_stuff():
    db_currency_init(cursor)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

