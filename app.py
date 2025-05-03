import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from gtts import gTTS
import matplotlib.pyplot as plt
import io
import uuid

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="💧 WaterWise Home", layout="wide")

# === LOAD TRAINED MODEL & SCALER ===
clf = joblib.load("classifier.pkl")
scaler = joblib.load("scaler.pkl")

# === LANGUAGE TOGGLE ===
lang = st.sidebar.radio("🌐 Language / Idioma", ("English", "Español"))

# === LABEL DEFINITIONS ===
labels = {
    "English": {
        "title": "💧 WaterWise Home: Save Water & Money",
        "intro": "Analyze water quality, estimate usage, and find cost-effective upgrades to save water and reduce bills.",
        "tabs": {
            "potability": "Water Portability",
            "usage": "Water Usage & Upgrades",
            "rebates": "Rebate Finder",
            "tips": "Conservation Tips"
        },
        "potability": {
            "intro": "Enter water quality metrics to predict **Portability** using a trained ML model.",
            "predict": "🔍 Predict",
            "result": ["✅ Drinkable", "❌ Not Drinkable"],
            "chart": "📊 Input Water Quality",
            "form": {
                "ph": "pH (0–14)",
                "Hardness": "Hardness (mg/L)",
                "Solids": "Solids (ppm)",
                "Chloramines": "Chloramines (ppm)",
                "Sulfate": "Sulfate (mg/L)",
                "Conductivity": "Conductivity (μS/cm)",
                "Organic_carbon": "Organic Carbon (ppm)",
                "Trihalomethanes": "Trihalomethanes (μg/L)",
                "Turbidity": "Turbidity (NTU)"
            }
        },
        "usage": {
            "title": "Estimate Your Water Usage",
            "intro": "Input your appliance details to estimate water usage and get upgrade recommendations.",
            "form": {
                "household_size": "Household Size (people)",
                "toilet_age": "Usage",
                "toilet_age_options": ["Pre-1992 (>3.5 GPF)", "1992–2000 (1.6 GPF)", "Post-2000 (≤1.28 GPF)"],
                "showerhead_flow": "Showerhead Flow Rate (GPM)",
                "faucet_flow": "Faucet Flow Rate (GPM)",
                "dishwasher_age": "Dishwasher Age",
                "dishwasher_age_options": ["Pre-1994", "1994–2010", "Post-2010 (ENERGY STAR)"],
                "washer_age": "Washing Machine Age",
                "washer_age_options": ["Pre-2000 (Top-load)", "2000–2010 (Top-load)", "Post-2010 (Front-load, ENERGY STAR)"],
                "water_heater": "Water Heater Type",
                "water_heater_options": ["Standard Tank", "Tankless", "Heat Pump"],
                "irrigation": "Irrigation System",
                "irrigation_options": ["None", "Standard Sprinklers", "Smart Controller"],
                "leak": "Suspected Leaks?",
                "leak_options": ["No", "Yes (Dripping faucet)", "Yes (Running toilet)"]
            },
            "calculate": "📈 Calculate Usage",
            "recommendations": "Recommended Upgrades",
            "savings_chart": "Cost-Benefit Analysis"
        },
        "rebates": {
            "title": "Find Rebates",
            "intro": "Enter your ZIP code to find water-saving rebates in your area.",
            "zip": "ZIP Code (e.g., 95207 for Stockton, CA)",
            "find": "🔎 Find Rebates"
        },
        "tips": {
            "title": "Water Conservation Tips",
            "intro": "Adopt these habits to save water and reduce your bills."
        },
        "tts_lang": "en"
    },
    "Español": {
        "title": "💧 Hogar WaterWise: Ahorra Agua y Dinero",
        "intro": "Analiza la calidad del agua, estima el uso y encuentra mejoras rentables para ahorrar agua y reducir facturas.",
        "tabs": {
            "potability": "Potabilidad del Agua",
            "usage": "Uso de Agua y Mejoras",
            "rebates": "Buscador de Reembolsos",
            "tips": "Consejos de Conservación"
        },
        "potability": {
            "intro": "Ingrese los parámetros de calidad del agua para predecir su **potabilidad** usando un modelo de ML.",
            "predict": "🔍 Predecir",
            "result": ["✅ Potable", "❌ No Potable"],
            "chart": "📊 Calidad del Agua Ingresada",
            "form": {
                "ph": "pH (0–14)",
                "Hardness": "Dureza (mg/L)",
                "Solids": "Sólidos (ppm)",
                "Chloramines": "Cloraminas (ppm)",
                "Sulfate": "Sulfato (mg/L)",
                "Conductivity": "Conductividad (μS/cm)",
                "Organic_carbon": "Carbono Orgánico (ppm)",
                "Trihalomethanes": "Trihalometanos (μg/L)",
                "Turbidity": "Turbidez (NTU)"
            }
        },
        "usage": {
            "title": "Estima tu Uso de Agua",
            "intro": "Ingresa los detalles de tus electrodomésticos para estimar el uso de agua y obtener recomendaciones de mejoras.",
            "form": {
                "household_size": "Tamaño del Hogar (personas)",
                "toilet_age": "Antigüedad del Inodoro",
                "toilet_age_options": ["Antes de 1992 (>3.5 GPF)", "1992–2000 (1.6 GPF)", "Después de 2000 (≤1.28 GPF)"],
                "showerhead_flow": "Tasa de Flujo de la Ducha (GPM)",
                "faucet_flow": "Tasa de Flujo del Grifo (GPM)",
                "dishwasher_age": "Antigüedad del Lavavajillas",
                "dishwasher_age_options": ["Antes de 1994", "1994–2010", "Después de 2010 (ENERGY STAR)"],
                "washer_age": "Antigüedad de la Lavadora",
                "washer_age_options": ["Antes de 2000 (Carga superior)", "2000–2010 (Carga superior)", "Después de 2010 (Carga frontal, ENERGY STAR)"],
                "water_heater": "Tipo de Calentador de Agua",
                "water_heater_options": ["Tanque Estándar", "Sin Tanque", "Bomba de Calor"],
                "irrigation": "Sistema de Riego",
                "irrigation_options": ["Ninguno", "Aspersores Estándar", "Controlador Inteligente"],
                "leak": "¿Fugas Sospechosas?",
                "leak_options": ["No", "Sí (Grifo goteando)", "Sí (Inodoro corriendo)"]
            },
            "calculate": "📈 Calcular Uso",
            "recommendations": "Mejoras Recomendadas",
            "savings_chart": "Análisis de Costo-Beneficio"
        },
        "rebates": {
            "title": "Busca Reembolsos",
            "intro": "Ingresa tu código postal para encontrar reembolsos para ahorrar agua en tu área.",
            "zip": "Código Postal (ej., 95207 para Stockton, CA)",
            "find": "🔎 Buscar Reembolsos"
        },
        "tips": {
            "title": "Consejos de Conservación de Agua",
            "intro": "Adopta estos hábitos para ahorrar agua y reducir tus facturas."
        },
        "tts_lang": "es"
    }
}
L = labels[lang]

# === HEADER ===
st.markdown(f"<h1 style='font-size:36px;'>{L['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='font-size:20px;'>{L['intro']}</p>", unsafe_allow_html=True)

# === TABS ===
tab1, tab2, tab3, tab4 = st.tabs([L["tabs"]["potability"], L["tabs"]["usage"], L["tabs"]["rebates"], L["tabs"]["tips"]])

# === TAB 1: WATER POTABILITY ===
with tab1:
    st.markdown(f"<h3>{L['potability']['intro']}</h3>", unsafe_allow_html=True)
    with st.form("water_form"):
        data = {
            "ph": st.number_input(L["potability"]["form"]["ph"], 0.0, 14.0, 7.0),
            "Hardness": st.number_input(L["potability"]["form"]["Hardness"], 0.0, 500.0, 150.0),
            "Solids": st.number_input(L["potability"]["form"]["Solids"], 0.0, 50000.0, 10000.0),
            "Chloramines": st.number_input(L["potability"]["form"]["Chloramines"], 0.0, 20.0, 5.0),
            "Sulfate": st.number_input(L["potability"]["form"]["Sulfate"], 0.0, 500.0, 250.0),
            "Conductivity": st.number_input(L["potability"]["form"]["Conductivity"], 0.0, 1000.0, 400.0),
            "Organic_carbon": st.number_input(L["potability"]["form"]["Organic_carbon"], 0.0, 30.0, 10.0),
            "Trihalomethanes": st.number_input(L["potability"]["form"]["Trihalomethanes"], 0.0, 120.0, 60.0),
            "Turbidity": st.number_input(L["potability"]["form"]["Turbidity"], 0.0, 10.0, 4.0)
        }
        submitted = st.form_submit_button(L["potability"]["predict"])

    if submitted:
        X = pd.DataFrame([data])
        X_scaled = scaler.transform(X)
        y_pred = clf.predict(X_scaled)[0]
        result_text = L["potability"]["result"][y_pred]
        st.markdown(f"<h2 style='font-size:28px; color: {'green' if y_pred == 1 else 'red'};'>{result_text}</h2>", unsafe_allow_html=True)
        plot_df = pd.DataFrame({"Feature": list(L["potability"]["form"].values()), "Value": list(data.values())})
        fig = px.bar(plot_df, x="Feature", y="Value", title=L["potability"]["chart"])
        st.plotly_chart(fig)
        tts = gTTS(text=result_text, lang=L["tts_lang"])
        tts.save("prediction.mp3")
        audio_file = open("prediction.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")

# === TAB 2: WATER USAGE & UPGRADES ===
with tab2:
    st.markdown(f"<h3>{L['usage']['title']}</h3>", unsafe_allow_html=True)
    st.markdown(L["usage"]["intro"])
   
    # Appliance data and efficiency metrics
    appliance_data = {
        "toilet": {"Pre-1992 (>3.5 GPF)": 5.0, "1992–2000 (1.6 GPF)": 1.6, "Post-2000 (≤1.28 GPF)": 1.28},
        "showerhead": {"standard": 2.5, "low-flow": 1.5},
        "faucet": {"standard": 2.2, "low-flow": 1.0},
        "dishwasher": {"Pre-1994": 10.0, "1994–2010": 6.0, "Post-2010 (ENERGY STAR)": 3.5},
        "washer": {"Pre-2000 (Top-load)": 40.0, "2000–2010 (Top-load)": 30.0, "Post-2010 (Front-load, ENERGY STAR)": 15.0},
        "water_heater": {"Standard Tank": 1.0, "Tankless": 0.9, "Heat Pump": 0.8},  # Efficiency multiplier
        "irrigation": {"None": 0, "Standard Sprinklers": 1000, "Smart Controller": 600}  # Gallons/month
    }
   
    with st.form("usage_form"):
        household_size = st.number_input(L["usage"]["form"]["household_size"], 1, 10, 4)
        toilet_age = st.selectbox(L["usage"]["form"]["toilet_age"], L["usage"]["form"]["toilet_age_options"])
        showerhead_flow = st.number_input(L["usage"]["form"]["showerhead_flow"], 0.0, 5.0, 2.5)
        faucet_flow = st.number_input(L["usage"]["form"]["faucet_flow"], 0.0, 5.0, 2.2)
        dishwasher_age = st.selectbox(L["usage"]["form"]["dishwasher_age"], L["usage"]["form"]["dishwasher_age_options"])
        washer_age = st.selectbox(L["usage"]["form"]["washer_age"], L["usage"]["form"]["washer_age_options"])
        water_heater = st.selectbox(L["usage"]["form"]["water_heater"], L["usage"]["form"]["water_heater_options"])
        irrigation = st.selectbox(L["usage"]["form"]["irrigation"], L["usage"]["form"]["irrigation_options"])
        leak = st.selectbox(L["usage"]["form"]["leak"], L["usage"]["form"]["leak_options"])
        submitted_usage = st.form_submit_button(L["usage"]["calculate"])

    if submitted_usage:
        # Calculate monthly water usage (gallons)
        toilet_usage = appliance_data["toilet"][toilet_age] * household_size * 5 * 30  # 5 flushes/day
        shower_usage = showerhead_flow * 5 * household_size * 30  # 5-minute showers
        faucet_usage = faucet_flow * 2 * household_size * 30  # 2 minutes/day
        dishwasher_usage = appliance_data["dishwasher"][dishwasher_age] * 4 * 4  # 4 loads/week
        washer_usage = appliance_data["washer"][washer_age] * 4 * 4  # 4 loads/week
        irrigation_usage = appliance_data["irrigation"][irrigation]
        leak_usage = 600 if "Dripping faucet" in leak else 4000 if "Running toilet" in leak else 0
        total_usage = (toilet_usage + shower_usage + faucet_usage + dishwasher_usage + washer_usage + irrigation_usage + leak_usage)
       
        # Compare to Stockton baseline (5,236 gallons)
        baseline = 5236
        bill_estimate = 33.24 + (total_usage / 748) * (49.11 - 33.24)  # Assuming similar tiered rate
       
        st.markdown(f"**Estimated Monthly Usage**: {total_usage:.0f} gallons")
        st.markdown(f"**Compared to Stockton Average (5,236 gallons)**: {'Above' if total_usage > baseline else 'Below'}")
        st.markdown(f"**Estimated Bill**: ${bill_estimate:.2f}")

        # Recommendations and savings calculations
        recommendations = []
        savings_list = []
        costs = []
        water_savings = []

        if toilet_age != "Post-2000 (≤1.28 GPF)":
            new_usage = 1.28 * household_size * 5 * 30
            water_saved = toilet_usage - new_usage
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**Upgrade Toilet**: Switch to a WaterSense toilet (1.28 GPF) to save ~{water_saved:.0f} gallons/month. Cost: ~$200, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(200)
            water_savings.append(water_saved)

        if showerhead_flow > 1.5:
            new_usage = 1.5 * 5 * household_size * 30
            water_saved = shower_usage - new_usage
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**Low-Flow Showerhead**: Install a 1.5 GPM showerhead to save ~{water_saved:.0f} gallons/month. Cost: ~$20, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(20)
            water_savings.append(water_saved)

        if faucet_flow > 1.0:
            new_usage = 1.0 * 2 * household_size * 30
            water_saved = faucet_usage - new_usage
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**Low-Flow Faucet Aerator**: Install a 1.0 GPM aerator to save ~{water_saved:.0f} gallons/month. Cost: ~$5, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(5)
            water_savings.append(water_saved)

        if dishwasher_age != "Post-2010 (ENERGY STAR)":
            new_usage = 3.5 * 4 * 4
            water_saved = dishwasher_usage - new_usage
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**ENERGY STAR Dishwasher**: Upgrade to save ~{water_saved:.0f} gallons/month. Cost: ~$500, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(500)
            water_savings.append(water_saved)

        if washer_age != "Post-2010 (Front-load, ENERGY STAR)":
            new_usage = 15.0 * 4 * 4
            water_saved = washer_usage - new_usage
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**ENERGY STAR Washer**: Upgrade to a front-load model to save ~{water_saved:.0f} gallons/month. Cost: ~$800, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(800)
            water_savings.append(water_saved)

        if water_heater != "Tankless":
            water_saved = 0.1 * (toilet_usage + shower_usage + faucet_usage)
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**Tankless Water Heater**: Switch to save ~{water_saved:.0f} gallons/month. Cost: ~$1000, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(1000)
            water_savings.append(water_saved)

        if irrigation != "Smart Controller" and irrigation != "None":
            new_usage = 600
            water_saved = irrigation_usage - new_usage
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**Smart Irrigation Controller**: Save ~{water_saved:.0f} gallons/month. Cost: ~$200, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(200)
            water_savings.append(water_saved)

        if leak != "No":
            water_saved = leak_usage
            cost_saved = water_saved * (49.11 - 33.24) / 748
            recommendations.append(f"**Fix Leaks**: Repair {leak.lower()} to save ~{water_saved:.0f} gallons/month. Cost: ~$50, Savings: ~${cost_saved:.2f}/month.")
            savings_list.append(cost_saved)
            costs.append(50)
            water_savings.append(water_saved)

        # Add greywater system and rainwater harvesting with savings
        water_saved = 1000  # Greywater system
        cost_saved = water_saved * (49.11 - 33.24) / 748
        recommendations.append(f"**Greywater System**: Reuse shower/laundry water for irrigation. Cost: ~$1500, Savings: ~{water_saved:.0f} gallons/month.")
        savings_list.append(cost_saved)
        costs.append(1500)
        water_savings.append(water_saved)

        water_saved = 500  # Rainwater harvesting
        cost_saved = water_saved * (49.11 - 33.24) / 748
        recommendations.append(f"**Rainwater Harvesting**: Collect rainwater for outdoor use. Cost: ~$500, Savings: ~{water_saved:.0f} gallons/month.")
        savings_list.append(cost_saved)
        costs.append(500)
        water_savings.append(water_saved)

        st.markdown(f"### {L['usage']['recommendations']}")
        for rec in recommendations:
            st.markdown(rec)

        # Cost-Benefit Chart
        if recommendations:  # Only plot if there are recommendations
            fig, ax = plt.subplots()
            annual_savings = [s * 12 for s in savings_list]  # Convert monthly to annual savings
            payback = [c / s if s > 0 else float('inf') for c, s in zip(costs, annual_savings)]
            ax.bar(range(len(recommendations)), payback)
            ax.set_xticks(range(len(recommendations)))
            ax.set_xticklabels([r.split("**")[1] for r in recommendations], rotation=45, ha="right")
            ax.set_ylabel("Payback Period (Years)")
            ax.set_title(L["usage"]["savings_chart"])
            st.pyplot(fig)

# === TAB 3: REBATE FINDER ===
with tab3:
    st.markdown(f"<h3>{L['rebates']['title']}</h3>", unsafe_allow_html=True)
    st.markdown(L["rebates"]["intro"])
    zip_code = st.text_input(L["rebates"]["zip"], "95207")
    if st.button(L["rebates"]["find"]):
        # Simulated rebate data (replace with API call in production)
        rebates = [
            {"item": "WaterSense Toilet", "amount": "$100", "source": "Cal Water"},
            {"item": "ENERGY STAR Washer", "amount": "$150", "source": "SoCal WaterSmart"},
            {"item": "Smart Irrigation Controller", "amount": "$200", "source": "LADWP"},
            {"item": "Low-Flow Showerhead", "amount": "$20", "source": "Valley Water"}
        ]
        st.markdown("**Available Rebates**")
        for rebate in rebates:
            st.markdown(f"- {rebate['item']}: {rebate['amount']} ({rebate['source']})")
        st.markdown("Check [SoCal WaterSmart](https://www.socalwatersmart.com/) or [Cal Water](https://www.calwater.com/) for details.")

# === TAB 4: CONSERVATION TIPS ===
with tab4:
    st.markdown(f"<h3>{L['tips']['title']}</h3>", unsafe_allow_html=True)
    st.markdown(L["tips"]["intro"])
    tips = [
        "Turn off the tap while brushing teeth to save up to 8 gallons/day.",
        "Take shorter showers (5 minutes or less) to save 12.5 gallons/shower.",
        "Run dishwashers and washers only with full loads to save 10–15 gallons/load.",
        "Fix leaks promptly; a dripping faucet can waste 600 gallons/month.",
        "Use a broom instead of a hose to clean driveways, saving 150 gallons/use.",
        "Install a rain barrel to collect rainwater for gardening, saving 500 gallons/month.",
        "Plant drought-tolerant native plants (xeriscaping) to reduce outdoor water use by 50%.",
        "Use a smart irrigation controller to optimize watering, saving up to 1000 gallons/month."
    ]
    for tip in tips:
        st.markdown(f"- {tip}")
    st.markdown("Source: [EPA WaterSense](https://www.epa.gov/watersense), [Energy.gov](https://www.energy.gov)")
