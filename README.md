# 🌍 LSTM CO₂ Emissions Forecasting: An SDG 13 Solution

## Project Goal
This project leverages Deep Learning to create an accurate forecasting tool for national carbon dioxide (CO₂) emissions. The objective is to provide policymakers with a dynamic, data-driven instrument to set and evaluate the effectiveness of climate mitigation strategies under **UN Sustainable Development Goal 13: Climate Action**.

## ⚙️ Machine Learning Approach
| Component         | Detail                                                                                                                                                            |
| ------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SDG Addressed** | SDG 13: Climate Action                                                                                                                                            |
| **ML Type**       | Supervised Learning (Time-Series Regression)                                                                                                                      |
| **Core Model**    | **Long Short-Term Memory (LSTM) Neural Network**                                                                                                                  |
| **Justification** | LSTMs excel at modeling sequential data, making them ideal for capturing complex, non-linear dependencies between annual economic indicators and CO₂ emissions over time. |

## 📊 Dataset & Features
The primary dataset, `SDG13_Climate_Action_Data.csv`, was compiled from global open-source climate and economic indicators (World Bank & Our World in Data sources).

| Feature Type    | Column Name                        | Role in Model                                            |
| ----------------| -----------------------------------|--------------------------------------------------------  |
| **Target (Y)**  | `CO2_Metric_Tons_Per_Capita`       | The variable the LSTM model is trained to predict.       |
| **Features (X)**| `Gas consumption per capita(mł)`   | Key predictor of future CO₂ output.                      |
| **Features (X)**| `Coal consumption per capita(Ton)` | Key predictor of future CO₂ output.                      |
| **Features (X)**| `Oil consumption per capita(mł)`   | Key predictor of future CO₂ output.                      |
| **Features (X)**| `Population`                       | Used for scaling and contextualizing per capita metrics. |

## 🚀 Model Performance and Demo

The model was trained using a look-back window of **10 years** of data to predict the next year's emissions.

**Final Evaluation Metric (Test Set):**
* **Mean Absolute Error (MAE):** ** 0.440 ** Metric Tons Per Capita

The MAE indicates that, on average, the model's prediction is highly accurate, providing a reliable forecast for informing policy.

### Results Visualization
Below is a plot comparing the model's predictions against the actual historical emissions in the test set.

<img width="1400" height="700" alt="co2_forecasting_results" src="https://github.com/user-attachments/assets/3de15705-c90e-466a-910d-be38cc49c780" />


## 💻 Repository Structure
| File                           | Description                                                                                                                                          |
| -------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                      | **The main executable code.** Contains data loading, normalization, sequence creation, the 2-layer LSTM definition, training, and final evaluation.  |
| `SDG13_Climate_Action_Data.csv`| The merged and cleaned dataset used for training the model.                                                                                          |
| `co2_forecasting_results.png`  | Output plot demonstrating the model's accuracy on unseen test data.                                                                                  |

## 💡 Ethical Reflection
The core of this solution is to promote **sustainability and fairness**. By focusing on *per capita* indicators and including a range of consumption features, the model minimizes bias that might result from training on raw, high-emission data primarily from large, developed nations. This allows developing countries to receive more equitable and tailored climate policy recommendations.

