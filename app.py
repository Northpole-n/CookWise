# app.py — FINAL, with model evaluation
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, re, ast
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try LightGBM if available
USE_LIGHTGBM = True
try:
    import lightgbm as lgb
except Exception:
    USE_LIGHTGBM = False
    lgb = None

# -----------------------------
# Sample recipes if recipes.csv missing
# -----------------------------
SAMPLE_RECIPES = [
    {
        "recipe_id": "r1",
        "name": "Avocado Toast",
        "ingredients": [{"ingredient_name": "avocado", "quantity": 50}, {"ingredient_name": "bread", "quantity": 2}],
        "prep_time_minutes": 8,
        "servings": 1,
        "nutrients": {"cal": 320, "protein": 6, "fat": 20, "carb": 30},
        "tags": ["breakfast", "vegetarian"]
    },
    {
        "recipe_id": "r2",
        "name": "Grilled Sandwich",
        "ingredients": [{"ingredient_name": "bread", "quantity": 2}, {"ingredient_name": "cheese", "quantity": 40}, {"ingredient_name": "tomato", "quantity": 40}],
        "prep_time_minutes": 10,
        "servings": 1,
        "nutrients": {"cal": 320, "protein": 12, "fat": 14, "carb": 35},
        "tags": ["snack", "breakfast"]
    },
    {
        "recipe_id": "r3",
        "name": "Omelette",
        "ingredients": [{"ingredient_name": "egg", "quantity": 2}, {"ingredient_name": "onion", "quantity": 30}, {"ingredient_name": "tomato", "quantity": 30}],
        "prep_time_minutes": 10,
        "servings": 1,
        "nutrients": {"cal": 200, "protein": 12, "fat": 8, "carb": 6},
        "tags": ["breakfast"]
    },
    {
        "recipe_id": "r4",
        "name": "Tomato Pasta",
        "ingredients": [{"ingredient_name": "pasta", "quantity": 100}, {"ingredient_name": "tomato", "quantity": 150}, {"ingredient_name": "cheese", "quantity": 50}],
        "prep_time_minutes": 20,
        "servings": 2,
        "nutrients": {"cal": 500, "protein": 18, "fat": 10, "carb": 70},
        "tags": ["dinner", "italian"]
    },
    {
        "recipe_id": "r5",
        "name": "Fried Rice",
        "ingredients": [{"ingredient_name": "rice", "quantity": 150}, {"ingredient_name": "egg", "quantity": 2}, {"ingredient_name": "soy_sauce", "quantity": 20}],
        "prep_time_minutes": 15,
        "servings": 2,
        "nutrients": {"cal": 450, "protein": 16, "fat": 8, "carb": 65},
        "tags": ["lunch", "asian"]
    },
    {
        "recipe_id": "r6",
        "name": "Greek Salad",
        "ingredients": [{"ingredient_name": "tomato", "quantity": 100}, {"ingredient_name": "onion", "quantity": 30}, {"ingredient_name": "cheese", "quantity": 30}],
        "prep_time_minutes": 10,
        "servings": 2,
        "nutrients": {"cal": 220, "protein": 8, "fat": 10, "carb": 12},
        "tags": ["salad", "vegetarian"]
    }
]

def ensure_recipes_file(path="recipes.csv"):
    if not os.path.exists(path):
        rows = []
        for r in SAMPLE_RECIPES:
            rows.append({
                "recipe_id": r["recipe_id"],
                "name": r["name"],
                "ingredients": str(r["ingredients"]),
                "prep_time_minutes": r.get("prep_time_minutes", 15),
                "servings": r.get("servings", 1),
                "nutrients": str(r["nutrients"]),
                "tags": str(r["tags"]),
                "difficulty": "easy"
            })
        pd.DataFrame(rows).to_csv(path, index=False)

# -----------------------------
# Load and parse recipes safely
# -----------------------------
@st.cache_data
def load_recipes(path="recipes.csv"):
    ensure_recipes_file(path)
    df = pd.read_csv(path)

    def safe_parse(x):
        if pd.isna(x):
            return {}
        if isinstance(x, (dict, list)):
            return x
        x = str(x).strip()
        try:
            return ast.literal_eval(x)
        except Exception:
            fixed = re.sub(r"(\w+):", r"'\1':", x)
            try:
                return ast.literal_eval(fixed)
            except Exception:
                return {}

    for col in ["ingredients", "nutrients", "tags"]:
        df[col] = df[col].apply(safe_parse)
    return df

# -----------------------------
# Compute recipe features
# -----------------------------
def compute_recipe_features(inventory, recipes):
    today = datetime.today()
    rows = []
    for _, r in recipes.iterrows():
        total_needed, available, missing, soonest = 0, 0, 0, 999
        for ing in r["ingredients"]:
            iname = ing["ingredient_name"]
            need = float(ing["quantity"])
            total_needed += need
            if iname in inventory:
                have = inventory[iname]["quantity"]
                exp = inventory[iname]["expiry"]
                if have >= need:
                    available += 1
                else:
                    missing += (need - have)
                try:
                    soonest = min(soonest, (exp - today).days)
                except Exception:
                    pass
            else:
                missing += need
        frac = available / max(1, len(r["ingredients"]))
        rows.append({
            "recipe_id": r["recipe_id"],
            "frac_avail": frac,
            "missing_qty": missing,
            "soonest_expiry_days": soonest if soonest != 999 else 30,
            "total_needed_grams": total_needed,
            "prep_time_minutes": r.get("prep_time_minutes", 15)
        })
    return pd.DataFrame(rows)

# -----------------------------
# Train model (returns model and evaluation info)
# -----------------------------
def train_model(X, y):
    """
    Trains a regression model (LightGBM if available else RandomForest).
    Returns: model, preds, dict(metrics)
    """
    if USE_LIGHTGBM and lgb is not None:
        data = lgb.Dataset(X, label=y)
        params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
        model = lgb.train(params, data, num_boost_round=100)
        preds = model.predict(X)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=1)
        model.fit(X, y)
        preds = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "y_true": y.values, "y_pred": preds}
    return model, metrics

# -----------------------------
# Utility: plot Predicted vs Actual
# -----------------------------
def plot_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.7)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)  # ideal line
    ax.set_xlabel("Actual (y)")
    ax.set_ylabel("Predicted (y_pred)")
    ax.set_title("Predicted vs Actual")
    plt.tight_layout()
    return fig

# -----------------------------
# Utility: feature importance plot (works for LightGBM and RandomForest)
# -----------------------------
def plot_feature_importance(model, feature_names):
    # try several ways to get importances
    importances = None
    if hasattr(model, "feature_importance"):  # LightGBM Booster or sklearn wrapper
        try:
            importances = model.feature_importance()
        except Exception:
            pass
    if importances is None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    if importances is None:
        return None

    ser = pd.Series(importances, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(5, 3))
    ser.plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance")
    plt.tight_layout()
    return fig

# -----------------------------
# Build exact plan (unchanged behavior) + show model eval
# -----------------------------
def build_plan(recipes, inventory, meals_per_day, days, pref="balanced", show_eval=False):
    feats = compute_recipe_features(inventory, recipes)
    X = feats[["frac_avail", "missing_qty", "soonest_expiry_days", "total_needed_grams", "prep_time_minutes"]]
    y = feats["frac_avail"] - 0.02 * feats["missing_qty"]

    model, metrics = train_model(X, y)

    # predictions for scoring
    preds = metrics["y_pred"]
    recipes = recipes.copy()
    recipes["score"] = preds

    # nutrition weighting
    def nutrition_bonus(n):
        if pref == "high protein":
            return n.get("protein", 0)
        if pref == "low carb":
            return -n.get("carb", 0)
        return 0

    recipes["score"] += recipes["nutrients"].apply(nutrition_bonus)
    recipes = recipes.sort_values("score", ascending=False).reset_index(drop=True)
    total_meals = int(days * meals_per_day)
    selected = [recipes.iloc[i % len(recipes)].to_dict() for i in range(total_meals)]

    # Compute missing ingredients per recipe
    plan, purchase = [], {}
    for d in range(days):
        day = []
        for s in range(meals_per_day):
            rec = selected[d * meals_per_day + s]
            miss = []
            for ing in rec["ingredients"]:
                n = ing["ingredient_name"]
                q = ing["quantity"]
                have = inventory.get(n, {"quantity": 0})["quantity"]
                if have < q:
                    miss.append(f"{n}: {q - have:.1f}")
                    purchase[n] = purchase.get(n, 0) + (q - have)
            rec["missing"] = miss
            day.append(rec)
        plan.append(day)

    # Optional evaluation outputs
    eval_outputs = {}
    eval_outputs["metrics"] = {"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]}
    eval_outputs["y_true"] = metrics["y_true"]
    eval_outputs["y_pred"] = metrics["y_pred"]
    eval_outputs["feature_names"] = X.columns.tolist()
    eval_outputs["model"] = model

    return plan, purchase, eval_outputs

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CookWise", layout="centered")
st.title("CookWise — Smart Leftover Meal Planner (with model evaluation)")

recipes = load_recipes()

if "inventory" not in st.session_state:
    st.session_state.inventory = {}

st.sidebar.header("Inventory")
with st.sidebar.form("add_ing"):
    n = st.text_input("Ingredient name")
    q = st.number_input("Quantity", min_value=0.0, value=100.0)
    e = st.date_input("Expiry", value=(datetime.today() + timedelta(days=5)).date())
    add = st.form_submit_button("Add")
    if add and n:
        st.session_state.inventory[n] = {"quantity": q, "expiry": datetime.combine(e, datetime.min.time())}
        st.sidebar.success(f"Added {n}")

if st.sidebar.button("Load sample inventory"):
    st.session_state.inventory = {
        "tomato": {"quantity": 200, "expiry": datetime.today() + timedelta(days=3)},
        "egg": {"quantity": 4, "expiry": datetime.today() + timedelta(days=2)},
        "onion": {"quantity": 100, "expiry": datetime.today() + timedelta(days=4)},
        "bread": {"quantity": 4, "expiry": datetime.today() + timedelta(days=3)},
        "rice": {"quantity": 300, "expiry": datetime.today() + timedelta(days=10)},
        "cheese": {"quantity": 100, "expiry": datetime.today() + timedelta(days=5)},
    }
    st.sidebar.success("Sample inventory loaded!")

st.subheader("Your Inventory")
if not st.session_state.inventory:
    st.info("No ingredients yet.")
else:
    st.dataframe(pd.DataFrame([
        {"Ingredient": k, "Quantity": v["quantity"], "Expiry": v["expiry"].date()}
        for k, v in st.session_state.inventory.items()
    ]))

st.markdown("---")
st.subheader("Plan Settings")
meals_per_day = st.number_input("Meals per day", 1, 5, 2)
days = st.number_input("Days to plan", 1, 7, 7)
pref = st.selectbox("Nutrition goal", ["balanced", "high protein", "low carb"])
show_eval = st.checkbox("Show model evaluation (RMSE, plots)", value=True)

if st.button("Generate Plan"):
    if not st.session_state.inventory:
        st.error("Add ingredients or load sample inventory.")
    else:
        plan, purchase, eval_outputs = build_plan(recipes, st.session_state.inventory, meals_per_day, days, pref, show_eval=show_eval)
        st.success(f"Generated {days}-day plan ({meals_per_day} meals/day)")

        # Display plan
        for d, meals in enumerate(plan):
            st.markdown(f"### Day {d+1}")
            for s, rec in enumerate(meals):
                name = rec["name"]
                n = rec["nutrients"]
                st.write(f" Slot {s+1} — {name}")
                st.caption(f"Calories: {n.get('cal',0)} kcal | Protein: {n.get('protein',0)}g | Carbs: {n.get('carb',0)}g")
                if rec["missing"]:
                    st.warning("Needs: " + ", ".join(rec["missing"]))
                else:
                    st.success("All ingredients available")

        st.markdown("---")
        st.subheader("Grocery Suggestions")
        if not purchase:
            st.write("No purchases needed!")
        else:
            dfp = pd.DataFrame(list(purchase.items()), columns=["Ingredient", "Qty to Buy"])
            st.table(dfp)
            dfp.to_csv("grocery_suggestion.csv", index=False)
            st.success("Saved grocery_suggestion.csv")

        # -----------------------------
        # Show model evaluation (metrics + plots)
        # -----------------------------
        if show_eval:
            st.markdown("---")
            st.subheader("Model Evaluation")

            metrics = eval_outputs["metrics"]
            st.write(f"**RMSE:** {metrics['rmse']:.4f}  |  **MAE:** {metrics['mae']:.4f}  |  **R²:** {metrics['r2']:.4f}")

            # Predicted vs Actual plot
            fig1 = plot_pred_vs_actual(eval_outputs["y_true"], eval_outputs["y_pred"])
            st.pyplot(fig1)

            # Feature importance
            fig2 = plot_feature_importance(eval_outputs["model"], eval_outputs["feature_names"])
            if fig2 is not None:
                st.pyplot(fig2)
            else:
                st.info("Feature importance not available for this model type.")

            # Optional: show a small table of actual vs predicted
            df_eval = pd.DataFrame({
                "recipe_id": compute_recipe_features(st.session_state.inventory, recipes)["recipe_id"],
                "actual_y": eval_outputs["y_true"],
                "predicted_y": np.round(eval_outputs["y_pred"], 4)
            })
            st.markdown("**Actual vs Predicted (per recipe):**")
            st.dataframe(df_eval)

