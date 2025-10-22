# Comprehensive Analysis Report

    ## Project Overview
    - Dataset size: 14640 tweets covering 6 major US airlines.
    - Primary objective: quantify airline sentiment, surface operational pain points, and power a predictive web dashboard.

    ## Data Quality Highlights
    - Duplicate rows detected: 36 (removed logically during preprocessing pipeline).
    - Columns with missing values: top gaps include `negativereason` (5462) and `tweet_coord` (13621).
    - Timestamp coverage: February 2015 with minute-level granularity enabling temporal analysis.

    ## Exploratory Insights
    - Negative sentiment share: 62.69% of tweets.
    - Highest tweet volume airline: United with 3822 mentions.
    - Most common negative reason: Customer Service Issue (2910 tweets).
    - Daily activity spans 3 aggregated sentiment records.

    ## Sentiment Pattern Discovery
    | antecedents        | consequents        |   support |   confidence |   lift |   leverage |   conviction |
|:-------------------|:-------------------|----------:|-------------:|-------:|-----------:|-------------:|
| airline=American   | sentiment=negative |  0.213554 |            1 |      1 |          0 |          inf |
| airline=Southwest  | sentiment=negative |  0.129222 |            1 |      1 |          0 |          inf |
| airline=Delta      | sentiment=negative |  0.104053 |            1 |      1 |          0 |          inf |
| airline=United     | sentiment=negative |  0.286882 |            1 |      1 |          0 |          inf |
| airline=US Airways | sentiment=negative |  0.246568 |            1 |      1 |          0 |          inf |

    ## Predictive Modeling
- Support Vector Machine (linear_svc) accuracy: 0.763. Artefacts stored in `models/linear_svc/` with confusion matrix at `visualizations/confusion_matrix_linear_svc.png`.
- Complement Naive Bayes accuracy: 0.759. Artefacts stored in `models/complement_nb/` with confusion matrix at `visualizations/confusion_matrix_complement_nb.png`.
- Comparative metrics are available in `reports/model_performance_linear_svc.md` and `reports/model_performance_complement_nb.md`, alongside CSV confusion matrices in `data/processed/confusion_matrix_linear_svc.csv` and `data/processed/confusion_matrix_complement_nb.csv`.

## Visualization Package
- Distribution overview: `visualizations/sentiment_distribution.png` plus interactive HTML counterpart.
- Airline comparison heatmap: `visualizations/airline_sentiment_heatmap.png`.
- Temporal trend: `visualizations/tweet_volume_timeline.png` and the daily sentiment overlay in `visualizations/daily_sentiment_trend.png`.
- Negative driver spotlight: `visualizations/top_negative_reasons.png` alongside the heatmap `visualizations/negative_reason_heatmap.png`.
- Sentiment share comparisons: `visualizations/sentiment_share_per_airline.png` and `visualizations/negative_ratio_per_airline.png`.
- Lexical focus: `visualizations/wordcloud_negative.png`, `visualizations/wordcloud_neutral.png`, and `visualizations/wordcloud_positive.png`.

    ## Key Takeaways
    1. Customer frustration is concentrated in delayed flights and customer service interactions.
    2. United Airlines and American Airlines receive the highest volume of complaints, primarily around delays and late arrivals.
    3. The linear SVM model delivers robust classification performance suitable for real-time prediction on the dashboard.
    4. Visual assets and structured analyses provide strong narrative support for stakeholder presentations.

    ## Next Steps
    - Integrate live Twitter streaming for real-time sentiment ingestion.
    - Expand model evaluation to include F1-optimized thresholds per sentiment class.
    - Conduct A/B testing on dashboard UI to validate usability for admin reviewers.
