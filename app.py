from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from flask import Flask, render_template, request, send_file


from flask import Flask, render_template, request


app = Flask(__name__)

# Ensure database folder exists
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "fittings.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# SQLite config
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# Paths
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "railway_fittings_dataset_2025.csv.xlsx")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "failure_model.pkl")

# ------------------------------
# DB MODELS
# ------------------------------
class Fitting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.String(80), unique=True, nullable=False)
    type = db.Column(db.String(80))
    vendor = db.Column(db.String(80))
    status = db.Column(db.String(20))  # Active / Failed
    supply_date = db.Column(db.DateTime)
    warranty_months = db.Column(db.Integer)
    location = db.Column(db.String(80))

class Inspection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fitting_id = db.Column(db.Integer, db.ForeignKey('fitting.id'))
    inspection_date = db.Column(db.DateTime, default=datetime.now)
    inspector_name = db.Column(db.String(50))
    result = db.Column(db.String(10))  # Pass / Fail
    next_inspection_date = db.Column(db.DateTime)

class Failure(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fitting_id = db.Column(db.Integer, db.ForeignKey('fitting.id'))
    failure_date = db.Column(db.DateTime, default=datetime.now)
    failure_type = db.Column(db.String(50))
    severity = db.Column(db.String(20))
    cost = db.Column(db.Float, default=0.0)

# ------------------------------
# Helper: preprocess dataset (returns processed df, X, y, encoders)
# ------------------------------
def preprocess_dataset(file_path, encoders=None, fit_encoders=False):
    """
    Loads dataset, creates target and features:
      - status (1 if Failure_Date exists else 0)
      - warranty_months (int parsed from 'Warranty_Period')
      - age_days (today - Date_of_Supply)
      - failures (0/1)  # from Failure_Date presence (single-row dataset)
    Categorical columns encoded: Item_Type, Vendor_Name, Location
    If encoders dict provided and fit_encoders=False -> use provided encoders for transform.
    If fit_encoders=True -> fit and return encoders.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_excel(file_path, parse_dates=['Date_of_Supply', 'Inspection_Date', 'Failure_Date'])
    df = df.copy()

    # Standardize column names (lower-case keys) to avoid simple mismatch. But we preserve original names too.
    # Expected original columns seen earlier: Item_ID, Item_Type, Vendor_Name, Lot_Number, Date_of_Supply,
    # Inspection_Date, Warranty_Period, Failure_Date, Location, Performance_Score
    # Create convenience columns with lower-case names for usage below if originals exist.

    # map expected names to normalized keys
    mapping = {}
    for c in df.columns:
        mapping[c.lower()] = c

    def col(name):
        return mapping.get(name.lower())

    # target: status
    failure_col = col('failure_date')
    if failure_col:
        df['status'] = df[failure_col].notna().astype(int)
        df['failures'] = df['status']  # single snapshot: 1 if has failure_date else 0
    else:
        df['status'] = 0
        df['failures'] = 0

    # warranty_months
    wcol = col('warranty_period')
    if wcol:
        # extract digits
        df['warranty_months'] = pd.to_numeric(df[wcol].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0).astype(int)
    else:
        df['warranty_months'] = 0

    # age_days
    supply_col = col('date_of_supply')
    if supply_col:
        today = datetime.now()
        df['age_days'] = (today - pd.to_datetime(df[supply_col])).dt.days.fillna(0).astype(int)
    else:
        df['age_days'] = 0

    # Performance score
    perf_col = col('performance_score')
    if perf_col:
        df['Performance_Score'] = pd.to_numeric(df[perf_col], errors='coerce').fillna(df[perf_col].mean() if df[perf_col].dtype != 'O' else 0)
    else:
        df['Performance_Score'] = 0

    # Lowercase item id normalization
    id_col = col('item_id') or col('item')
    if id_col:
        df['item_id_norm'] = df[id_col].astype(str)
    else:
        df['item_id_norm'] = df.index.astype(str)

    # Categorical columns to encode
    cat_cols = []
    if col('item_type'):
        df['Item_Type'] = df[col('item_type')].astype(str)
        cat_cols.append('Item_Type')
    else:
        df['Item_Type'] = 'unknown'

    if col('vendor_name'):
        df['Vendor_Name'] = df[col('vendor_name')].astype(str)
        cat_cols.append('Vendor_Name')
    else:
        df['Vendor_Name'] = 'unknown'

    if col('location'):
        df['Location'] = df[col('location')].astype(str)
        cat_cols.append('Location')
    else:
        df['Location'] = 'unknown'

    # Fit or apply label encoders
    encs = {} if encoders is None else encoders
    if fit_encoders:
        for c in cat_cols:
            le = LabelEncoder()
            df[c + "_enc"] = le.fit_transform(df[c])
            encs[c] = le
    else:
        for c in cat_cols:
            if c in encs and isinstance(encs[c], LabelEncoder):
                # transform; for unseen labels this will raise — we catch and map unseen to a new value index (len(classes))
                le = encs[c]
                values = df[c].astype(str).tolist()
                transformed = []
                for v in values:
                    if v in le.classes_:
                        transformed.append(int(le.transform([v])[0]))
                    else:
                        # unseen -> add a placeholder index len(classes) (model wasn't trained on this but handle gracefully)
                        transformed.append(len(le.classes_))
                df[c + "_enc"] = transformed
            else:
                # fallback to fitting new encoder (not ideal, but ensures route doesn't break)
                le = LabelEncoder()
                df[c + "_enc"] = le.fit_transform(df[c].astype(str))
                encs[c] = le

    # Decide final feature set
    features = ['age_days', 'warranty_months', 'Performance_Score', 'Item_Type_enc', 'Vendor_Name_enc', 'Location_enc']

    # ensure encoded columns present
    for c in ['Item_Type_enc','Vendor_Name_enc','Location_enc']:
        if c not in df.columns:
            df[c] = 0

    X = df[features].fillna(0)
    y = df['status'].astype(int)

    return df, X, y, encs, features

# ------------------------------
# TRAIN ROUTE
# ------------------------------
# ------------------------------
# TRAIN ROUTE (with dataset selection)
# ------------------------------
@app.route('/train')
def train_route():
    try:
        # check if user passed ?file=<dataset>
        file = request.args.get("file", None)
        if file:
            file_path = os.path.join(DATA_DIR, file)
            if not os.path.exists(file_path):
                return f"❌ Dataset {file} not found in {DATA_DIR}"
        else:
            file_path = DATA_FILE  # default dataset

        # preprocess selected dataset
        df, X, y, encoders, features = preprocess_dataset(file_path, encoders=None, fit_encoders=True)

        # train/test split and model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # save model + encoders + features
        payload = {
            "model": model,
            "encoders": encoders,
            "features": features,
            "dataset": file if file else "default"
        }
        joblib.dump(payload, MODEL_FILE)

        return f"✅ Model trained on {os.path.basename(file_path)} and saved ({MODEL_FILE}). Accuracy: {acc:.3f}"
    except Exception as e:
        return f"Error training model: {e}"

# ------------------------------
# PREDICT FOR ENTIRE DATASET (render table)
# ------------------------------
@app.route('/predict_all')
def predict_all_route():
    try:
        if not os.path.exists(MODEL_FILE):
            return "Model not found. Please visit /train first."

        saved = joblib.load(MODEL_FILE)
        model = saved['model']
        encoders = saved['encoders']
        features = saved['features']

        df, X, y, _enc, _ = preprocess_dataset(DATA_FILE, encoders=encoders, fit_encoders=False)

        probs = model.predict_proba(X)[:, 1]

        results = []
        for i, p in enumerate(probs):
            if p < 0.33:
                risk = "Low"
            elif p < 0.66:
                risk = "Medium"
            else:
                risk = "High"
            results.append({
                "item_id": df.iloc[i]['item_id_norm'],
                "probability": round(float(p), 3),
                "risk": risk
            })

        # optionally show aggregated counts on analytics page
        counts = {"Low": 0, "Medium": 0, "High": 0}
        for r in results:
            counts[r['risk']] += 1

        return render_template("predict.html", results=results, counts=counts)

    except Exception as e:
        return f"Error predicting dataset: {e}"

# ------------------------------
# PREDICT SINGLE ITEM (by Item ID present in dataset)
# ------------------------------
@app.route('/predict_item/<string:item_id>')
def predict_item_route(item_id):
    try:
        if not os.path.exists(MODEL_FILE):
            return "Model not found. Please visit /train first."

        saved = joblib.load(MODEL_FILE)
        model = saved['model']
        encoders = saved['encoders']
        features = saved['features']

        df_raw = pd.read_excel(DATA_FILE, parse_dates=['Date_of_Supply', 'Inspection_Date', 'Failure_Date'])

        # Find the correct Item_ID column
        id_col = None
        for c in df_raw.columns:
            if c.lower() == 'item_id':
                id_col = c
                break
        if id_col is None:
            return "Dataset does not contain Item_ID column."

        row = df_raw[df_raw[id_col].astype(str).str.lower() == item_id.lower()]
        if row.empty:
            return f"Item ID {item_id} not found in dataset."

        # Preprocess single-row DataFrame
        row_df = row.copy()
        temp_path = os.path.join("data", "temp_single_row.xlsx")
        row_df.to_excel(temp_path, index=False)
        df_proc, X, y, _e, _f = preprocess_dataset(temp_path, encoders=encoders, fit_encoders=False)
        os.remove(temp_path)

        prob = model.predict_proba(X)[:, 1][0]
        pred = model.predict(X)[0]

        if prob < 0.33:
            risk = "Low"
        elif prob < 0.66:
            risk = "Medium"
        else:
            risk = "High"

        return render_template(
            "predict_item.html",
            item_id=item_id,
            probability=round(float(prob), 3),
            prediction=int(pred),
            risk=risk,
            details=row.to_dict(orient="records")[0]  # send original row details
        )

    except Exception as e:
        return f"Error predicting item: {e}"
# ------------------------------
# ANALYTICS (simple page showing counts & link to predictions)
# ------------------------------
@app.route('/analytics')
def analytics():
    # If model exists, show counts of risk levels; otherwise render a simple analytics page
    try:
        if os.path.exists(MODEL_FILE):
            saved = joblib.load(MODEL_FILE)
            encoders = saved['encoders']
            df, X, y, _e, _f = preprocess_dataset(DATA_FILE, encoders=encoders, fit_encoders=False)
            model = saved['model']
            probs = model.predict_proba(X)[:, 1]
            counts = {"Low":0, "Medium":0, "High":0}
            for p in probs:
                if p < 0.33:
                    counts["Low"] += 1
                elif p < 0.66:
                    counts["Medium"] += 1
                else:
                    counts["High"] += 1
            total = len(probs)
            return render_template("analytics.html", counts=counts, total=total)
        else:
            return render_template("analytics.html", counts=None, total=0)
    except Exception as e:
        return f"Error loading analytics: {e}"

# ------------------------------
# Homepage - uses DB counts (if DB empty this still works)
# ------------------------------
@app.route('/')
def index():
    try:
        active_fittings = Fitting.query.filter_by(status='Active').count()
        failed_fittings = Fitting.query.filter_by(status='Failed').count()
        overdue_inspections = Inspection.query.filter(Inspection.next_inspection_date < datetime.now()).count()
        recent_failures = Failure.query.filter(Failure.failure_date >= datetime.now() - timedelta(days=30)).count()
    except Exception:
        # if DB not ready, fallback to zeros
        active_fittings = failed_fittings = overdue_inspections = recent_failures = 0

    return render_template('index.html',
                           active_fittings=active_fittings,
                           failed_fittings=failed_fittings,
                           overdue_inspections=overdue_inspections,
                           recent_failures=recent_failures)
# ------------------------------
# QR CODE MANAGEMENT
# ------------------------------
@app.route('/qr_codes')
def qr_codes():
    try:
        fittings = Fitting.query.all()
    except Exception:
        fittings = []
    return render_template("qr_codes.html", fittings=fittings)


# ------------------------------
# SCANNER PAGE
# ------------------------------
@app.route('/scan')
def scan():
    return render_template("scanner.html")


# ------------------------------
# REPORTS PAGE
# ------------------------------
@app.route("/reports")
def reports():
    try:
        df = pd.read_excel("railway_fittings_dataset_2025.csv.xlsx")

        # ✅ Parse date columns safely
        for col in ["Date_of_Supply", "Inspection_Date", "Failure_Date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # --- Filters from request ---
        date_from = request.args.get("date_from")
        date_to = request.args.get("date_to")
        status = request.args.get("status")
        vendor = request.args.get("vendor")

        # --- Apply filters ---
        if date_from:
            try:
                date_from = datetime.strptime(date_from, "%Y-%m-%d")
                df = df[df["Date_of_Supply"] >= date_from]
            except Exception as e:
                print("Date_from filter error:", e)

        if date_to:
            try:
                date_to = datetime.strptime(date_to, "%Y-%m-%d")
                df = df[df["Date_of_Supply"] <= date_to]
            except Exception as e:
                print("Date_to filter error:", e)

        if status and "Status" in df.columns:
            df = df[df["Status"] == status]

        if vendor and "Vendor_Name" in df.columns:
            df = df[df["Vendor_Name"] == vendor]

        # --- Prepare lists for template ---
        vendors = sorted(df["Vendor_Name"].dropna().unique().tolist()) if "Vendor_Name" in df.columns else []
        reports = df.to_dict(orient="records")

        # ✅ Only send first 200 rows + total count
        return render_template(
            "report.html",
            reports=reports[:200],
            vendors=vendors,
            total_reports=len(reports)
        )

    except Exception as e:
        # ✅ show real error in browser
        import traceback
        traceback.print_exc()
        return f"<h3 style='color:red;'>Error loading reports: {e}</h3>"

@app.route("/")
def dashboard():
    # Dummy values (replace later with Excel or DB)
    stats = {
        "active_fittings": 120,
        "failed_fittings": 15,
        "overdue_inspections": 8,
        "recent_failures": 5
    }
    return render_template("index.html", **stats)

if __name__ == "__main__":
    app.run(debug=True)
# ------------------------------
# DATABASE TEMPLATE + FILE MGMT
# ------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

@app.route("/database")
def database_page():
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith((".csv", ".xlsx"))]
    except Exception:
        files = []
    return render_template("database.html", files=files)

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if "dataset" not in request.files:
        return "No file part"
    file = request.files["dataset"]
    if file.filename == "":
        return "No selected file"

    save_path = os.path.join(DATA_DIR, file.filename)
    file.save(save_path)
    return f"✅ Uploaded {file.filename}. <a href='/database'>Back</a>"

@app.route("/preview_dataset/<string:filename>")
def preview_dataset(filename):
    try:
        file_path = os.path.join(DATA_DIR, filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        preview_html = df.head(20).to_html(classes="table table-bordered table-sm", index=False)
        return f"<h3>Preview: {filename}</h3>{preview_html}<br><a href='/database'>Back</a>"
    except Exception as e:
        return f"Error previewing dataset: {e}"

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
