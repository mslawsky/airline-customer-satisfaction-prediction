# Airline Customer Satisfaction Analysis 🛫

This repository contains a comprehensive analysis of airline customer satisfaction factors using multiple machine learning approaches. Our models identify key drivers of satisfaction to guide strategic investments in customer experience.

![Airline Customer Satisfaction](inflight-entertainment-rating.png)

---

## Project Overview 📊

This analysis employs logistic regression, decision tree, and random forest models to identify and validate the key factors driving airline customer satisfaction. Using survey data from 129,880 customers, we provide data-driven insights to guide operational decisions and resource allocation.

---

## Key Findings 🔍

### Primary Driver of Satisfaction

All models consistently identified **in-flight entertainment** as the most significant predictor of customer satisfaction:

- **Logistic Regression**: Demonstrated the strong correlation between entertainment ratings and satisfaction probability
- **Decision Tree**: Quantified that in-flight entertainment accounts for approximately 46% of the impact on satisfaction prediction
- **Random Forest**: Confirmed in-flight entertainment as the top feature while providing more robust feature importance rankings

### Feature Importance Ranking

Our decision tree model clearly identified the hierarchical importance of different features:

![Feature Importance](features-ranked.png)

### Secondary Factors of Importance

The decision tree analysis revealed additional important factors:
- **Seat comfort** (20% importance)
- **Ease of online booking** (8% importance)
- **Customer type** and **class of travel** (smaller but notable impacts)

### Critical Thresholds

Our analysis identified key satisfaction thresholds:
- Entertainment ratings between 2-3 represent the critical transition zone
- Customers rating entertainment 4-5 have >70% probability of overall satisfaction
- Customers rating entertainment below 2 have <20% probability of satisfaction

### Decision Tree Visualization

The visualization below shows how the model makes decisions, with in-flight entertainment as the top-level split:

![Decision Tree Tuned](decision-tree-hypertuned.png)

---

## Models and Performance 📈

### Logistic Regression Model
- **Accuracy**: 80.15%
- **Precision**: 81.61%
- **Recall**: 82.15%
- **F1 Score**: 81.88%

### Decision Tree Model
- **Accuracy**: 94.02%
- **Precision**: 95.41%
- **Recall**: 93.55%
- **F1 Score**: 94.47%

### Random Forest Model
- **Accuracy**: 95.28%
- **Precision**: 96.13% 
- **Recall**: 94.89%
- **F1 Score**: 95.50%

The confusion matrices show excellent classification performance, with the random forest model offering slight improvements over the decision tree due to its ensemble approach.

![Confusion Matrix](confusion-matrix.png)

The higher performance of both tree-based models suggests that customer satisfaction involves non-linear relationships and interaction effects between variables that the logistic regression model cannot fully capture.

### Data Leakage Identification & Resolution

During our random forest implementation, we identified and addressed a critical data leakage issue:

- **The Issue**: One-hot encoding created two binary columns from our target variable: `satisfaction_satisfied` (our target) and `satisfaction_dissatisfied` (its inverse). Including the latter in our feature set created data leakage.

- **The Impact**: Initially, this resulted in artificially perfect model performance metrics (all 1.0000), as the model was effectively using the target variable to predict itself.

- **The Solution**: We explicitly removed both satisfaction-related columns from the feature set and used only `satisfaction_satisfied` as our target, ensuring honest model evaluation.

This experience highlights the importance of careful feature selection and the potential pitfalls of automated preprocessing in machine learning workflows.

---

## Strategic Recommendations 💡

Based on our multi-model analysis, we recommend:

1. **Prioritize in-flight entertainment improvements**
   - Focus on bringing the entertainment experience from "average" (2-3) to "good" (4+)
   - This represents the highest ROI opportunity for satisfaction improvement

2. **Invest in seat comfort enhancements**
   - As the second most important factor, seat ergonomics and quality improvements will yield significant satisfaction gains

3. **Streamline online booking processes**
   - Optimizing the digital booking experience can drive meaningful satisfaction increases

4. **Implement predictive satisfaction modeling**
   - Deploy our high-accuracy model (95%+) to identify potentially dissatisfied customers before journey completion
   - Enable proactive service recovery opportunities

5. **Adopt ensemble methods for production**
   - Consider implementing the random forest model in production environments for its superior generalization capabilities and robustness to outliers

---

## Repository Contents 📁

1. **Data Files**
   - [Invisto Airline Data](invisto-airline.csv) (CSV)

2. **Python Scripts**
   - [Airline Logistic Regression](airline-cs-logistic.py) (PY)
   - [Airline Decision-Tree Analysis](airline-cs-decision-tree.py) (PY)
   - [Airline Random Forest](airline-cs-random-forest.py) (PY)

3. **Visualizations**
   - [Confusion Matrix](confusion-matrix.png) (PNG)
   - [Hyper-tuned Decision Tree](decision-tree-hypertuned.png) (PNG)
   - [Decision Tree](decision-tree.png) (PNG)
   - [Ranked Features](features-ranked.png) (PNG)
   - [Inflight Entertainment Rating](inflight-entertainment-rating.png) (PNG)

---

## Model Evolution 🧠

This project demonstrates a progression of increasingly sophisticated modeling approaches:

1. **Logistic Regression**: Provided a baseline model with good interpretability but limited ability to capture non-linear relationships.

2. **Decision Tree**: Significantly improved performance by capturing non-linear patterns and feature interactions, with high interpretability.

3. **Random Forest**: Further refined our predictive capabilities through ensemble learning, offering the highest performance and robustness against overfitting.

Each model added value to our understanding, with the random forest serving as the most powerful predictor while maintaining the interpretability advantages of tree-based methods.

---

## Future Work 🚀

Potential extensions of this analysis include:
- Additional ensemble methods such as gradient boosting for even higher accuracy
- Time-series analysis to track satisfaction changes after service improvements
- Cluster analysis to identify distinct customer segments with different satisfaction drivers
- Deployment of a real-time satisfaction prediction system
- Neural network approaches for handling complex interaction effects

---

## Contact Information ✉️

For inquiries about this analysis:
- [LinkedIn Profile](https://www.linkedin.com/in/melissaslawsky/)
- [Client Results](https://melissaslawsky.com/portfolio/)
- [Tableau Portfolio](https://public.tableau.com/app/profile/melissa.slawsky1925/vizzes)
- [Email](mailto:melissa@melissaslawsky.com)

---

© Melissa Slawsky 2025. All Rights Reserved.
