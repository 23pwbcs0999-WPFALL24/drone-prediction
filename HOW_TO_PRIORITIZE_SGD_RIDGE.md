# ✅ HOW TO PRIORITIZE SGD+RIDGE MODEL

## **3 SIMPLE STEPS:**

### **Step 1: Modify Cell 5 (SAVE THE MODEL)**
After running Cell 4, add this code to Cell 5:

```python
# Save all 3 components for production
joblib.dump(ridge_model, 'drone_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly_transformer.pkl')

print("✅ Saved:")
print("   - drone_model.pkl (Ridge)")
print("   - scaler.pkl (StandardScaler)")
print("   - poly_transformer.pkl (PolynomialFeatures)")
```

---

### **Step 2: SKIP or COMMENT OUT Cell 10 (Random Forest)**
❌ **Don't run Cell 10** for production use  
(It's educational, but Random Forest won't work on 26M rows)

Instead, add this comment:
```python
# Cell 10 - SKIPPED FOR PRODUCTION
# Random Forest is not suitable for 26M row dataset
# Using SGD + Ridge model instead (Cell 4)
```

---

### **Step 3: DONE! ✅**
Your app.py is already updated to use:
- ✅ `drone_model.pkl` (Ridge Regressor)
- ✅ `scaler.pkl` (StandardScaler)  
- ✅ `poly_transformer.pkl` (PolynomialFeatures)

---

## **RUN ORDER:**
```
Cell 1  → Load data
Cell 2  → Markdown
Cell 3  → Feature selection
Cell 4  → SGD Convergence + Ridge Training ⭐
Cell 5  → SAVE MODEL (with step above) ⭐
Cell 6  → Markdown
Cell 7  → Markdown
Cell 8  → Markdown  
Cell 9  → Display sample data
Cell 10 → SKIP (Random Forest - not for 26M rows)
Cell 11 → Display sample data
Cell 12 → Model comparison (optional, for reference)
Cell 13 → Noise testing (optional)
Cell 14 → Generate app.py ✅ (already updated)
Cell 15 → Deploy with: streamlit run app.py
```

---

## **THEN RUN:**
```powershell
cd d:\dataset
python -m streamlit run app.py
```

---

## **SUMMARY:**
✅ Cell 4-5: Train SGD+Ridge, save components  
❌ Cell 10: Skip Random Forest  
✅ app.py: Uses SGD+Ridge (already updated)  
✅ Production-ready for 26M rows!
