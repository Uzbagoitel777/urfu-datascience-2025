import streamlit as st
import requests
from random import randint
from datetime import datetime, date

# Page configuration
st.set_page_config(
    page_title="Kickstarter Campaign Analyzer",
    page_icon="🚀",
    layout="wide"
)

# IP = '176.108.250.114'
IP = 'localhost'
port = '8000'

# Main title
st.title("🚀 Kickstarter Campaign Parameter Setup")
st.markdown("Set your campaign parameters and choose analysis models")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Campaign Parameters")

    # Campaign Name
    name = st.text_input(
        "Campaign Name",
        placeholder="Enter your campaign name",
        help="The name of your Kickstarter campaign"
    )

    # Category and Main Category
    category = st.text_input(
        "Category",
        placeholder="e.g., Technology, Games, Design",
        help="Specific category of your campaign"
    )

    main_category = st.text_input(
        "Main Category",
        placeholder="e.g., Technology, Creative, etc.",
        help="Broader category classification"
    )

    # Currency - Using selectbox with common currencies
    currency_options = [
        "USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "SEK", "NOK", "DKK",
        "PLN", "CZK", "HUF", "RON", "BGN", "HRK", "RUB", "CNY", "INR", "BRL",
        "MXN", "KRW", "SGD", "HKD", "NZD", "ZAR", "TRY", "ILS", "AED", "SAR"
    ]

    currency = st.selectbox(
        "Currency",
        options=currency_options,
        index=0,  # Default to USD
        help="Select the currency for your campaign"
    )

    # Date inputs
    col_date1, col_date2 = st.columns(2)

    with col_date1:
        launch_date = st.date_input(
            "Launch Date",
            value=date.today(),
            help="When will your campaign launch?"
        )

    with col_date2:
        deadline_date = st.date_input(
            "Deadline Date",
            value=date.today(),
            help="When will your campaign end?"
        )

    # Goal slider
    goal = st.slider(
        "Funding Goal",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=50.0,
        format="%.2f",
        help="Set your funding goal amount"
    )

    # Country
    country = st.text_input(
        "Country",
        placeholder="e.g., US, UK, CA, DE",
        help="Country code where the campaign will be launched"
    )

with col2:
    st.header("Analysis Models")
    st.markdown("Choose which models to use for analysis:")

    # Model selection checkboxes
    logistic_regression = st.checkbox(
        "Logistic Regression",
        value=True,
        help="Statistical model for binary classification"
    )

    decision_tree = st.checkbox(
        "Decision Tree",
        value=False,
        help="Tree-based model for classification"
    )

    neural_network = st.checkbox(
        "Neural Network (MLP)",
        value=False,
        help="Multi-layer perceptron neural network"
    )

    st.markdown("---")

    # Display current selections
    st.subheader("Current Selection Summary")
    if name:
        st.write(f"**Campaign:** {name}")
    if category:
        st.write(f"**Category:** {category}")
    if goal:
        st.write(f"**Goal:** {currency} {goal:,.2f}")

    selected_models = []
    if logistic_regression:
        selected_models.append("Logistic Regression")
    if decision_tree:
        selected_models.append("Decision Tree")
    if neural_network:
        selected_models.append("Neural Network")

    if selected_models:
        st.write(f"**Models:** {', '.join(selected_models)}")

# Analysis button and API call
st.markdown("---")
col_button, col_status = st.columns([1, 2])

with col_button:
    analyze_button = st.button(
        "🔍 Analyze Campaign",
        type="primary",
        help="Send parameters to analysis API"
    )


def validate_inputs():
    """Validate all required inputs"""
    errors = []

    if not name.strip():
        errors.append("Campaign name is required")
    if not category.strip():
        errors.append("Category is required")
    if not main_category.strip():
        errors.append("Main category is required")
    if not country.strip():
        errors.append("Country is required")
    if launch_date >= deadline_date:
        errors.append("Deadline must be after launch date")
    if not any([logistic_regression, decision_tree, neural_network]):
        errors.append("At least one analysis model must be selected")

    return errors


def prepare_api_payload(faked_data=True):
    """Prepare the JSON payload for the API"""
    launch_formatted = datetime.combine(launch_date, datetime.min.time()).isoformat()
    deadline_formatted = datetime.combine(deadline_date, datetime.min.time()).isoformat()

    payload = {
        'models': {
            'Logistic': logistic_regression,
            'Tree': decision_tree,
            'MLP': neural_network
        },
        'parameters': {
            # 'ID': randint(1001000000, 9999999999),
            'name': name.strip(),
            'category': category.strip(),
            'main_category': main_category.strip(),
            'currency': currency,
            'launched': launch_formatted,
            'deadline': deadline_formatted,
            'goal': float(goal),
            'country': country.strip(),
            'state': 'live',
            'backers': 0,
            'pledged': 0,
            'usd_pledged': 0,
            'usd_pledged_real': 0,
            'usd_goal_real': 0
        }
    }

    if faked_data:
        payload = fake_data(payload)

    return payload


def fake_data(payload):
    data = payload['parameters']
    data['backers'] = 105
    dur = int(data['deadline'][:4]) - int(data['launched'][:4])
    data['launched'] = str(2010 + randint(0, 8)) + data['launched'][4:]
    data['deadline'] = str(2010 + randint(0, 8) + dur) + data['deadline'][4:]

    payload['parameters'] = data

    return payload



def call_fastapi_endpoint(payload):
    """Call the FastAPI endpoint with the payload"""
    try:
        api_url = f"http://{IP}:{port}/analyze"

        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )

        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the analysis API. Make sure the FastAPI server is running."
    except requests.exceptions.Timeout:
        return None, "Request to the analysis API timed out."
    except requests.exceptions.HTTPError as e:
        return None, f"API returned an error: {e.response.status_code}"
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"


# Handle the analyze button click
if analyze_button:
    with col_status:
        # Validate inputs
        validation_errors = validate_inputs()

        if validation_errors:
            st.error("Please fix the following errors:")
            for error in validation_errors:
                st.write(f"• {error}")
        else:
            # Prepare payload
            payload = prepare_api_payload()

            # Show payload for debugging (optional)
            with st.expander("View API Payload", expanded=True):
                st.json(payload)

            # Show loading spinner
            with st.spinner("Analyzing campaign parameters..."):
                # Call the API
                result, error = call_fastapi_endpoint(payload)

                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Analysis completed successfully!")

                    # Display results
                    # if result:
                    #     st.subheader("Analysis Results")
                    #     st.json(result)

                    if result:
                        st.subheader("Analysis Results")

                        MODELS = {
                            "Logistic Regression": ("Logistic_Success", "Logistic_Prob"),
                            "Decision Tree": ("Tree_Success", "Tree_Prob"),
                            "Neural Network (MLP)": ("MLP_Success", "MLP_Prob"),
                        }

                        for pretty_name, (succ_key, prob_key) in MODELS.items():
                            success = result.get(succ_key)
                            prob = result.get(prob_key)

                            # Skip inactive models (both values are None)
                            if success is None and prob is None:
                                continue

                            # Human‑friendly strings
                            verdict_icon = "✅" if success else "❌"
                            verdict_txt = "Success" if success else "Failure"
                            prob_txt = f"{prob * 100:.2f} %" if prob is not None else "N/A"

                            # Render
                            st.markdown(f"""
                    <div style="padding:0.4rem 0">
                      <span style="font-size:1.1rem;font-weight:600">{pretty_name}</span><br>
                      &nbsp;&nbsp;{verdict_icon}&nbsp;<b>Verdict:</b> {verdict_txt}<br>
                      &nbsp;&nbsp;📊&nbsp;<b>Success probability:</b> {prob_txt}
                    </div>
                    <hr style="margin:0.3rem 0;">
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Kickstarter Campaign Analyzer | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
