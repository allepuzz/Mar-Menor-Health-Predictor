
### Hey!! 👋

This was my end-of-degree proyect!

## TL;DR

I built a predictor for concentrations of several micronutrients to try to, one understand what was happening to my childhood's beach lagoon and two, try to know its future's health. 

#### --Technical--
Time series cross-validation changes everything. Expanding window prevents data leakage. Bootstrap sampling breaks temporal order. Lag features solve RF limitations. GridSearchCV found optimal hyperparameters. 256 splits minimized validation error. Random Forest outperformed SARIMA consistently. Feature engineering beat model complexity. MSE dropped from 5.622 to 1.211. Supervised structure preserved temporal dependencies. Loss curves revealed no overfitting. Multiple metrics validated robustness. Seasonal patterns emerged naturally. Model generalized beyond training data.

#### --Real-World--
Real projects are messy. Time series data is tricky. Random Forest beat specialized models. Hyperparameter tuning cuts errors dramatically. Real data needs compromises. Eight iterations taught more than theory. Perfect is the enemy of done. Engineering meets environmental science. Personal motivation drives better work. High errors aren't always bad. Validate everything multiple times. Interdisciplinary work is valuable. Limitations are okay to admit.

### WHOLE SPILL:

# Mar Menor Water Quality Prediction using Machine Learning

A predictive modeling study I developed for my final degree project, analyzing the chemical and physicochemical state of the Mar Menor lagoon using Machine Learning algorithms.

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Motivation](#motivation)
- [What I Built](#what-i-built)
- [Technologies I Used](#technologies-i-used)
- [How I Approached It](#how-i-approached-it)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [What I'd Do Next](#what-id-do-next)
- [About Me](#about-me)
- [Acknowledgments](#acknowledgments)

## 🌊 About the Project

For my Bachelor's Thesis, I developed a predictive model using Machine Learning to estimate and predict critical variables that affect the Mar Menor, Europe's largest saltwater lagoon located in my home region of Murcia, Spain.

I used physicochemical, biological, and environmental data collected from 2019 to present to predict three key water quality indicators:
- **Chlorophyll-α concentration** (shows phytoplankton biomass)
- **Nitrate concentration (NO₃)**
- **Phosphate concentration (PO₄)**

These variables are the main indicators of water quality according to Spanish Royal Decree 817/2015.

### The Problem: Eutrophication

The Mar Menor has been facing serious ecological challenges due to eutrophication - basically too many nutrients in the water, mainly from agricultural runoff. This causes:
- Excessive algae and phytoplankton growth
- Murky water that blocks sunlight
- Lower oxygen levels
- Harm to the ecosystem's biodiversity

## 💡 Motivation

I was born in Cartagena, so the Mar Menor has always been part of my life. Over the years, I've watched it deteriorate, and places I used to enjoy became less pleasant to visit. This transformation really affected me and became a big driver for choosing this topic for my thesis.

I wanted to use what I learned during my engineering degree to understand this real, current problem that's close to home. While I know one person's work can't reverse all the damage, I hoped that through my background in engineering and computer science, I could contribute to better understanding the current state of the lagoon and where it might be heading under certain pressures.

My goal was to add modestly to the body of knowledge that environmental managers could use to make informed decisions.

## ✨ What I Built

- **Time Series Cross-Validation**: I implemented expanding window cross-validation to respect the temporal nature of the data
- **Model Comparison**: I compared Random Forest, Linear Regression, and SARIMA models to find the best performer
- **Hyperparameter Optimization**: I used GridSearchCV to find the optimal model configuration
- **Future Predictions**: The model generates forecasts up to 2027
- **Compliance Check**: I compared predictions against legal water quality thresholds

## 🛠 Technologies I Used

- **Python 3.x**
- **Machine Learning Libraries**:
  - scikit-learn (Random Forest, Linear Regression, GridSearchCV)
  - statsmodels (SARIMA)
- **Data Processing**:
  - pandas
  - numpy
- **Visualization**:
  - matplotlib
- **Environment**: Virtual environment (.venv)

### Hardware
I ran everything on my personal laptop:
- 16 GB RAM
- Intel Core i7-1280p (12th Gen, 2 GHz)

## 📊 How I Approached It

### 1. Gathering the Data

I combined data from two main sources:
- **Universidad Politécnica de Cartagena's Scientific Data Server**
  - Variables: temperature, salinity, transparency, chlorophyll, oxygen, turbidity
  - 1,044 measurements from August 2019 onwards
  
- **Canal Mar Menor Foundation**
  - Nitrate and phosphate concentrations
  - Flow rate and conductivity
  - Data from 12 measurement buoys across the lagoon

### 2. Building the Model (8 Iterations)

I went through several iterations to improve the model:

1. **First attempt**: Simple chlorophyll-α detection with limited features
2. **Second iteration**: First try at predicting all three target variables
3. **Third iteration**: Added more variables to the dataset (conductivity, flow rate)
4. **Fourth iteration**: Made the target variables dependent on each other during prediction
5. **Fifth iteration**: Optimized the number of data splits (found 256 splits worked best)
6. **Sixth iteration**: Fine-tuned Random Forest hyperparameters using GridSearchCV
7. **Seventh iteration**: Compared my Random Forest model against Linear Regression and SARIMA
8. **Final iteration**: Implemented future value predictions

### 3. Final Model Configuration

After extensive testing, my best Random Forest configurations were:
- **Chlorophyll-α**: 100 estimators, max_depth=20, bootstrap=True
- **Nitrates**: 100 estimators, max_depth=10, bootstrap=False
- **Phosphates**: 200 estimators, max_depth=20, bootstrap=False

**Key decisions I made**:
- Used 256 splits for time series cross-validation
- Created lag features (n_lags=12) to capture temporal patterns
- Structured the problem so predictions only use past values

## 📈 Results

### Model Performance (Mean Squared Error)

After comparing three different approaches, here's what I found:

| Variable | My Random Forest | Linear Regression | SARIMA |
|----------|--------------|-------------------|--------|
| **Chlorophyll-α** | **1.211** | 5.316 | 3.671 |
| **Nitrates** | **252.867** | 470.022 | 852.426 |
| **Phosphates** | **0.016** | 0.194 | 0.056 |

My Random Forest model outperformed the other approaches across all three variables, which was encouraging!

### Water Quality Assessment

I compared my predictions against the legal thresholds from Royal Decree 817/2015:

| Variable | Legal Threshold | My Average Prediction | Status |
|----------|-----------|-------------------|---------|
| Chlorophyll-α | 1.8 µg/L | 1.91 µg/L | ⚠️ Slightly above |
| Nitrates | 0.080 mg/L | 125.24 mg/L | ❌ Way above threshold |
| Phosphates | 0.072 mg/L | 0.22 mg/L | ⚠️ Above acceptable |

### What I Learned

1. **The model works**: My Random Forest implementation successfully captures temporal patterns in the water quality data
2. **Nitrates are a serious concern**: The predictions consistently show very high nitrate concentrations, which reflects the ongoing eutrophication problem
3. **Seasonal patterns matter**: The model picks up on seasonal variations in water quality
4. **Good generalization**: I checked the loss curves and didn't see signs of overfitting, which was a relief

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.x
Virtual environment capability
```

### Setup
```bash
# Clone the repository
git clone [your-repo-url]
cd mar-menor-ml-prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
```bash
# Execute main prediction script
python main.py

# For specific variable prediction
python predict_chlorophyll.py
python predict_nitrates.py
python predict_phosphates.py
```

## 🔮 What I'd Do Next

There are several things I'd like to explore if I continue this work:

1. **Try Deep Learning**: Compare with RNN/LSTM models - I focused on traditional ML but deep learning could be interesting
2. **Add Confidence Intervals**: Implement prediction intervals for the forecasts (like SARIMA has)
3. **Test More Algorithms**: Try Gradient Boosting Decision Trees - I considered them but focused on Random Forest
4. **More Features**: Incorporate weather data and information about nearby agricultural activities
5. **Real-time Dashboard**: Build something that could monitor water quality continuously
6. **Intervention Modeling**: Simulate what different remediation strategies might achieve

## 🙏 Acknowledgments

This project wouldn't have happened without several people:

- **My thesis advisor, Gonzalo Martínez Muñoz** - His work on the Tagus River inspired this entire project, and he trusted me with this idea from the start
- **Data providers**: Universidad Politécnica de Cartagena and Canal Mar Menor Foundation for making the data publicly available
- **My friends** - both from home and those I met in Madrid - for keeping me motivated on tough days
- **My siblings, cousins, and grandparents** - for being the best examples of love and unconditional support I could ask for
- **My parents** - most importantly, for their incredible patience with me, for their time and attention, and for giving me the opportunity to become who I am today

## 📄 License

I developed this project as my Bachelor's Thesis at Universidad Autónoma de Madrid. All rights are reserved according to the university's intellectual property regulations.

If you're interested in using this work or collaborating, feel free to reach out!

---

## 🎓 What I Learned

### Technical Skills
**Machine Learning in Practice**

I got hands-on experience with Random Forest, Linear Regression, and SARIMA - not just theoretically but actually implementing and comparing them
I learned that what works in theory doesn't always translate directly to real problems (like discovering RF outperformed even specialized time series models like SARIMA)
I understood the importance of hyperparameter tuning - the difference between my initial model (MSE of 5.622 for chlorophyll) and optimized one (MSE of 1.211) was huge

**Working with Time Series Data**

I learned why you can't just use standard cross-validation with temporal data - the order matters!
I discovered the challenge of using Random Forest with time series (the bootstrap sampling issue) and how to work around it with lag features
I got comfortable with the concept of "walking forward" through data to validate predictions

**Real-World Data is Messy**

I had to merge two different databases with different start dates
I made tough decisions about missing data (like treating phosphates as null before 2021)
I learned that sometimes you have to work with what you have, not what would be ideal

**Model Evaluation Beyond Accuracy**

I learned to use multiple metrics (MSE, MAE, R²) to really understand model performance
I discovered how to check for overfitting using loss curves
I understood that a high error (like 252.867 for nitrates) doesn't necessarily mean the model is bad - it might just reflect highly variable real-world data

### Domain Knowledge
**Environmental Science Basics**

I learned about eutrophication - what it is, what causes it, and why it's devastating
I now understand the biochemical indicators of water quality (chlorophyll-α, nutrients, dissolved oxygen)
I learned about the regulatory framework for water quality in Spain

**The Mar Menor Ecosystem**

I gained deep knowledge about a local environmental crisis that I'd witnessed but never fully understood
I learned about the complex interplay between agriculture, urban development, and marine ecosystems
I discovered how political, economic, and environmental factors all intersect in this issue

### Project Management & Research Skills
**Iterative Development**

I learned that good projects evolve - I went through 8 major iterations, each teaching me something new
I discovered the importance of testing assumptions (like comparing my RF model against alternatives)
I learned when to stop optimizing and when to keep pushing

**Working Independently**

I managed a multi-month project mostly on my own
I learned to set milestones and evaluate my progress
I got better at knowing when to ask for help (acknowledging my advisor's guidance)

**Interdisciplinary Thinking**

I learned to communicate between two very different fields - telecommunications engineering and marine biology
I got comfortable reading scientific papers outside my core discipline
I learned to respect the limits of my expertise (acknowledging I'm not a marine biologist)

### Problem-Solving & Critical Thinking
**Making Trade-offs**

I chose RF over Deep Learning despite current trends - and learned to justify that decision based on data characteristics
I decided against implementing some features (like confidence intervals) due to time/complexity constraints
I learned when "good enough" is actually good enough

**Dealing with Constraints**

I worked within the limits of my personal laptop (not cloud computing)
I managed with publicly available data (couldn't collect my own)
I learned to be creative within constraints rather than being paralyzed by limitations

**Scientific Skepticism**

I learned to validate my results against multiple baselines
I questioned whether my good results were too good (checking for overfitting)
I compared against specialized models (SARIMA) to make sure I wasn't missing something obvious

### Soft Skills
**Writing for Different Audiences**

I learned to explain complex technical concepts clearly
I got better at structuring a long-form technical document
I practiced balancing technical depth with readability
---

## 📚 Key References

Some papers and documents that were particularly helpful:
- Real Decreto 817/2015 (Spanish Water Quality Standards)
- Zhang et al. (2017) - "State-of-the-art classification algorithms"
- Bergmeir & Benítez (2012) - "Cross-validation for time series"
- Breiman (2001) - "Random Forests"

The full thesis document has a complete bibliography if you're interested.

---
Thank you so much for reading!

I'll leave you with my favourite quote:

*"Men lie, women lie, numbers don't" - Lil B, the Based God*

(I included this quote in my thesis because at the end of the day, the data tells the real story)


