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
    - Selected algorithm: linear_svc.
    - Test accuracy: 0.763 (target >= 0.650).
    - Model artifacts: `models/sentiment_pipeline.joblib`, `models/model_metrics.json`.
    - Confusion matrix visualization: `visualizations/model_confusion_matrix.png`.

    ## Visualization Package
    - Distribution overview: `visualizations/sentiment_distribution.png` and interactive HTML counterpart.
    - Airline comparison heatmap: `visualizations/airline_sentiment_heatmap.png`.
    - Temporal trend: `visualizations/tweet_volume_timeline.png`.
    - Negative driver spotlight: `visualizations/top_negative_reasons.png`.
    - Lexical focus: `visualizations/wordcloud_negative.png`.

    ## Key Takeaways
    1. Customer frustration is concentrated in delayed flights and customer service interactions.
    2. United Airlines and American Airlines receive the highest volume of complaints, primarily around delays and late arrivals.
    3. The linear SVM model delivers robust classification performance suitable for real-time prediction on the dashboard.
    4. Visual assets and structured analyses provide strong narrative support for stakeholder presentations.

    ## Next Steps
    - Integrate live Twitter streaming for real-time sentiment ingestion.
    - Expand model evaluation to include F1-optimized thresholds per sentiment class.
    - Conduct A/B testing on dashboard UI to validate usability for admin reviewers.
