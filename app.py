

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import joblib
# import lime.lime_tabular
# import shap
# import matplotlib.pyplot as plt
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Ensure 'static' directory exists
# os.makedirs("static", exist_ok=True)

# # Load trained model & training data
# xgb_model = joblib.load("xgb_model.pkl")
# X_train = joblib.load("X_train.pkl")  # Load training data for LIME
# feature_columns = X_train.columns.tolist()

# # Initialize LIME Explainer
# explainer = lime.lime_tabular.LimeTabularExplainer(
#     training_data=X_train.values,
#     feature_names=X_train.columns.values,
#     discretize_continuous=False,
#     class_names=["legit", "fraud"],
#     mode="classification",
#     verbose=True,
#     random_state=45
# )

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         try:
#             # Get user input from form (comma-separated values)
#             user_input = request.form["features"].split(",")
#             user_data = [float(value.strip()) for value in user_input]
#             user_sample = pd.DataFrame([user_data], columns=feature_columns)
            
#             # Predict fraud probability
#             fraud_probability = xgb_model.predict_proba(user_sample)[0][1]
#             threshold = 0.5
#             prediction = "FRAUD" if fraud_probability > threshold else "NOT FRAUD"
            
#             # Generate LIME Explanation
#             exp = explainer.explain_instance(user_sample.iloc[0], xgb_model.predict_proba)
#             lime_path = "static/lime_explanation.png"
#             fig = exp.as_pyplot_figure()
#             fig.savefig(lime_path, bbox_inches='tight', dpi=300)
#             plt.close(fig)
            
#             # Save detailed LIME explanation as HTML
#             lime_html_path = "static/lime_explanation.html"
#             with open(lime_html_path, "w", encoding="utf-8") as f:
#                 f.write(exp.as_html())

#             # SHAP Explanations
#             explainer_shap = shap.TreeExplainer(xgb_model)
#             shap_values = explainer_shap(user_sample)

#             # SHAP Summary Plot
#             shap_summary_path = "static/shap_summary.png"
#             shap.summary_plot(shap_values, user_sample, feature_names=feature_columns, show=False)
#             plt.savefig(shap_summary_path, bbox_inches='tight', dpi=300)
#             plt.close()

#             # SHAP Waterfall Plot
#             shap_waterfall_path = "static/shap_waterfall.png"
#             shap.plots.waterfall(shap_values[0], show=False)
#             plt.savefig(shap_waterfall_path, bbox_inches='tight', dpi=300)
#             plt.close()

#             # SHAP Force Plot (Saved as HTML)
#             shap_force_path = "static/shap_force.html"
#             shap_html = shap.force_plot(
#                 explainer_shap.expected_value, 
#                 shap_values.values[0], 
#                 user_sample.iloc[0]
#             )
#             shap.save_html(shap_force_path, shap_html)

#             return render_template(
#                 "result.html", 
#                 prediction=prediction, 
#                 probability=fraud_probability,
#                 lime_path=lime_path, 
#                 lime_html_path=lime_html_path,
#                 shap_summary_path=shap_summary_path,
#                 shap_waterfall_path=shap_waterfall_path,
#                 shap_force_path=shap_force_path
#             )

#         except Exception as e:
#             return render_template("index.html", error=str(e))
    
#     return render_template("index.html", feature_columns=feature_columns)

# if __name__ == "__main__":
#     app.run(debug=True)









import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure 'static' directory exists
os.makedirs("static", exist_ok=True)

# Load trained model & training data
xgb_model = joblib.load("xgb_model.pkl")
X_train = joblib.load("X_train.pkl")  # Load training data for LIME
feature_columns = X_train.columns.tolist()

# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.values,
    discretize_continuous=True,
    class_names=["legitimate", "fraud"],
    mode="classification",
    verbose=True,
    random_state=45
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            
            user_input = request.form["features"].split(",")
            user_data = [float(value.strip()) for value in user_input]
            user_sample = pd.DataFrame([user_data], columns=feature_columns)
            
            fraud_probability = xgb_model.predict_proba(user_sample)[0][1]
            threshold = 0.5
            prediction = "FRAUD" if fraud_probability > threshold else "NOT FRAUD"
    
            exp = explainer.explain_instance(user_sample.iloc[0], xgb_model.predict_proba)
            lime_path = "static/lime_explanation.png"
            fig = exp.as_pyplot_figure()
            fig.savefig(lime_path, bbox_inches='tight', dpi=300)
            plt.close(fig)

            lime_html_path = "static/lime_explanation.html"
            with open(lime_html_path, "w", encoding="utf-8") as f:
                f.write(exp.as_html())

            explainer_shap = shap.TreeExplainer(xgb_model)
            shap_values = explainer_shap(user_sample)

            shap_summary_path = "static/shap_summary.png"
            shap.summary_plot(shap_values, user_sample, feature_names=feature_columns, show=False)
            plt.savefig(shap_summary_path, bbox_inches='tight', dpi=300)
            plt.close()

            shap_waterfall_path = "static/shap_waterfall.png"
            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig(shap_waterfall_path, bbox_inches='tight', dpi=300)
            plt.close()

            shap_force_path = "static/shap_force.html"
            shap_html = shap.force_plot(
                explainer_shap.expected_value, 
                shap_values.values[0], 
                user_sample.iloc[0]
            )
            shap_html = f"<head>{shap.getjs()}</head><body>{shap_html.html()}</body>"
            with open(shap_force_path, "w", encoding="utf-8") as f:
                f.write(shap_html)

            return render_template(
                "result.html", 
                prediction=prediction, 
                probability=fraud_probability,
                lime_path=lime_path, 
                lime_html_path=lime_html_path,
                shap_summary_path=shap_summary_path,
                shap_waterfall_path=shap_waterfall_path,
                shap_force_path=shap_force_path
            )

        except Exception as e:
            return render_template("index.html", error=str(e))
    
    return render_template("index.html", feature_columns=feature_columns)

if __name__ == "__main__":
    app.run(debug=True)
