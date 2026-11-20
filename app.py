from __future__ import annotations
from datetime import date, timedelta
import json
import joblib
import pandas as pd
import streamlit as st
from config import DATA_PATH, EDA_DIR, METRICS_PATH, MODEL_PATH, TARGET
from src.predict import predict_service

st.set_page_config(page_title="CarCare AI | Car Service Planner", page_icon="🚘", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} .block-container{padding-top:1.4rem;max-width:1320px}
[data-testid="stMetric"]{background:#f6f8fb;border:1px solid #e6eaf0;padding:16px;border-radius:14px}
.hero{padding:28px 30px;border-radius:18px;color:#f8fbff;background:linear-gradient(115deg,#112a3b,#154d59 65%,#27816f);margin-bottom:22px}
.hero h1{font-family:Manrope,sans-serif;font-size:2.1rem;margin:4px 0 8px}.hero p{color:#c5e0e3;margin-bottom:0}
.notice{background:#fff8e8;border-left:4px solid #e0a538;padding:12px 15px;border-radius:8px;color:#654c14;font-size:.92rem}
.small-muted{color:#768494;font-size:.88rem}
</style>
""", unsafe_allow_html=True)

if "service_history" not in st.session_state:
    st.session_state.service_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None


def render_home():
    st.markdown("<div class='hero'><div>CARCARE AI · PASSENGER CARS ONLY</div><h1>Know your next service window.</h1><p>A practical service estimate grounded in your odometer, recent use, and the interval from your owner's manual.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='notice'><b>Prototype notice:</b> the ML model is trained on reproducible synthetic examples. Estimates demonstrate the software workflow and are not validated against real vehicle records or a mechanical inspection. Follow the car owner's manual and seek a qualified technician for vehicle concerns.</div>", unsafe_allow_html=True)
    st.write("")
    a,b,c=st.columns(3)
    with a:
        st.subheader("01 · Enter car details")
        st.write("Add the current reading, last service and the intervals specified in the owner's manual.")
    with b:
        st.subheader("02 · Estimate service window")
        st.write("A saved regression pipeline estimates remaining kilometres. Manual distance and time limits cap the result.")
    with c:
        st.subheader("03 · Review and record")
        st.write("See an approximate date, service status, model evidence and session-only prediction history.")
    st.divider()
    st.subheader("What this project demonstrates")
    st.write("Synthetic dataset generation, exploratory analysis, preprocessing pipelines, Decision Tree / Random Forest / Extra Trees regression, validation-based model selection, test-set reporting, and a Streamlit user interface.")
    st.caption("Service intervals are inputs: this application does not assume every car follows the same schedule.")


def render_vehicle():
    st.title("Car information")
    st.caption("Enter values for a passenger car. Use the interval printed in its owner's manual.")
    with st.form("vehicle_form"):
        first, second = st.columns(2)
        with first:
            brand=st.text_input("Car brand", placeholder="e.g., Honda")
            model_name=st.text_input("Car model", placeholder="e.g., Civic")
            year=st.number_input("Manufacturing year", min_value=1980, max_value=date.today().year, value=max(1980,date.today().year-5), step=1)
            current=st.number_input("Current odometer (km)", min_value=0, max_value=2000000, value=48000, step=500)
            last=st.number_input("Odometer at last service (km)", min_value=0, max_value=2000000, value=40000, step=500)
            service_date=st.date_input("Date of last service", value=date.today()-timedelta(days=180), max_value=date.today())
        with second:
            daily=st.number_input("Average daily distance (km)", min_value=0.1, max_value=2000.0, value=35.0, step=1.0)
            driving=st.selectbox("Typical driving condition", ["City","Mixed","Highway"])
            interval_km=st.number_input("Manual-recommended service interval (km)", min_value=1000, max_value=50000, value=10000, step=500)
            interval_months=st.number_input("Manual-recommended service interval (months)", min_value=1, max_value=60, value=12, step=1)
            service_type=st.selectbox("Most recent service category", ["Oil and filter","Inspection","Major service"])
            st.caption("If the owner's manual specifies other values, enter those instead.")
        submitted=st.form_submit_button("Estimate next service", type="primary", use_container_width=True)
    if submitted:
        values={"brand":brand,"model":model_name,"manufacturing_year":int(year),"current_odometer_km":float(current),
                "last_service_odometer_km":float(last),"last_service_date":service_date,"average_daily_km":float(daily),
                "driving_condition":driving,"recommended_interval_km":float(interval_km),
                "recommended_interval_months":int(interval_months),"previous_service_type":service_type}
        model_obj=load_model()
        if model_obj is None:
            st.error("Trained model not found. Run `python -m src.train` from the project folder, then refresh the app.")
        else:
            try:
                result=predict_service(values,model_obj)
                record={**values,**result,"created_at":date.today().isoformat()}
                st.session_state.last_prediction=record
                st.session_state.service_history.insert(0,record)
            except ValueError as exc:
                st.error(str(exc))
    result=st.session_state.last_prediction
    if result:
        st.divider()
        st.subheader(f"Service estimate · {result['brand']} {result['model']}")
        k1,k2,k3=st.columns(3)
        k1.metric("Estimated remaining distance",f"{result['remaining_km']:,} km")
        k2.metric("Approximate time window",f"{result['remaining_days']} days")
        k3.metric("Estimated service date",result["service_date"].strftime("%d %b %Y"))
        status=result["status"]
        if status=="Service Due": st.error(f"**{status}** · Arrange service soon and consult your owner's manual.")
        elif status=="Service Approaching": st.warning(f"**{status}** · Plan a service appointment.")
        else: st.success(f"**{status}** · Continue monitoring the odometer and service interval.")
        st.info(result["explanation"])
        with st.expander("How the estimate was constrained"):
            st.write(f"Model estimate before manual cap: **{result['model_estimated_remaining_km']:,} km**")
            st.write(f"Distance remaining by owner's manual: **{result['manual_distance_limit_km']:,} km**")
            st.write(f"Time remaining by owner's manual: **{result['manual_time_limit_days']:,} days**")
            st.caption("Status thresholds: Due at 5% or less of configured interval or within 30 days; Approaching at 15% or less or within 90 days; otherwise Not Due. A zero limit is Due.")


def render_dashboard():
    st.title("Dashboard")
    records=st.session_state.service_history
    m=load_metrics()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Session predictions",len(records))
    c2.metric("Cars recorded",len({(r["brand"],r["model"]) for r in records}))
    c3.metric("Selected model",m["selected_model"] if m else "Not trained")
    c4.metric("Held-out test rows",m["test_rows"] if m else "—")
    st.subheader("Model comparison")
    if m:
        frame=pd.DataFrame(m["model_metrics"]).T.rename_axis("Model").reset_index()
        st.dataframe(frame.style.format({"MAE":"{:.1f}","MSE":"{:.1f}","RMSE":"{:.1f}","R2":"{:.3f}"}),use_container_width=True,hide_index=True)
        st.caption("Metrics are computed on held-out synthetic test records. They are not estimates of real-world accuracy.")
        if (EDA_DIR/"07_model_comparison.png").exists(): st.image(str(EDA_DIR/"07_model_comparison.png"),caption="Held-out test MAE by model (lower is better)",use_container_width=True)
    else:
        st.info("Model metrics will appear after you run `python -m src.train`.")
    if records:
        st.subheader("Current session vehicle usage")
        rows=[{"Car":f"{r['brand']} {r['model']}","Odometer (km)":int(r["current_odometer_km"]),"Since service (km)":int(r["current_odometer_km"]-r["last_service_odometer_km"]),"Daily km":r["average_daily_km"],"Driving":r["driving_condition"],"Status":r["status"]} for r in records]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else:
        st.caption("Prediction totals are based on this browser session only; no sample counts are prefilled.")


def render_eda():
    st.title("Data and model analysis")
    st.caption("Plots below are produced from the reproducible synthetic dataset included with the project.")
    if DATA_PATH.exists():
        data=pd.read_csv(DATA_PATH)
        st.write(f"Dataset rows: **{len(data):,}** · Columns: **{len(data.columns)}** · Missing cells: **{int(data.isna().sum().sum())}** · Duplicate rows: **{int(data.duplicated().sum())}**")
    names=[("01_odometer_distribution.png","Vehicle mileage distribution"),("02_service_intervals.png","Service interval distribution"),
           ("03_age_service_frequency.png","Vehicle age and service frequency"),("04_distance_since_service.png","Distance since last service"),
           ("05_condition_maintenance.png","Driving conditions and maintenance patterns"),("06_correlation_heatmap.png","Numeric feature correlation heatmap")]
    for i in range(0,len(names),2):
        left,right=st.columns(2)
        for column,(filename,caption) in zip([left,right],names[i:i+2]):
            path=EDA_DIR/filename
            with column:
                if path.exists(): st.image(str(path),caption=caption,use_container_width=True)
                else: st.info("Run `python -m src.train` to generate EDA figures.")


def render_history():
    st.title("Service history")
    st.caption("Records are kept in Streamlit session memory and disappear when this session resets. No database is configured.")
    rows=st.session_state.service_history
    if not rows:
        st.info("Predictions you make during this session will appear here.")
        return
    table=[{"Car":f"{r['brand']} {r['model']}","Prediction date":r["created_at"],"Last service odometer (km)":r["last_service_odometer_km"],
            "Current odometer (km)":r["current_odometer_km"],"Estimated remaining (km)":r["remaining_km"],"Estimated due date":r["service_date"].isoformat(),"Status":r["status"]} for r in rows]
    st.dataframe(pd.DataFrame(table),use_container_width=True,hide_index=True)
    if st.button("Clear session history"):
        st.session_state.service_history=[]; st.session_state.last_prediction=None; st.rerun()


pages={"Home":render_home,"Vehicle information and prediction":render_vehicle,"Dashboard":render_dashboard,"Data and model analysis":render_eda,"Service history":render_history}
with st.sidebar:
    st.title("🚘 CarCare AI")
    st.caption("Smart car service prediction")
    selected=st.radio("Navigation",list(pages),label_visibility="collapsed")
    st.divider()
    st.caption("Passenger cars only · Prototype ML estimate")
pages[selected]()
