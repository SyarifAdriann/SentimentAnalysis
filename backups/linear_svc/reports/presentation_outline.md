# Presentation Outline (15-20 minutes)

## Slide 1 - Title & Motivation
- Airline Customer Sentiment on Twitter: Insights & Predictive Dashboard
- Motivation: social listening for service recovery.

## Slide 2 - Dataset Overview
- 14.6k tweets, 6 major US airlines, February 2015 snapshot.
- Key fields: text, sentiment, negative reason, timestamp.

## Slide 3 - Data Quality Check
- Missing value profile and duplicates.
- Actions taken during preprocessing.

## Slide 4 - Sentiment Landscape
- Negative sentiment at 62.69% dominates customer perception.
- Visual: sentiment_distribution.png (bar chart).

## Slide 5 - Airline Benchmarking
- Tweet volume per airline plus heatmap of sentiment counts.
- Highlight airlines with highest complaint ratios.

## Slide 6 - Pain Point Deep Dive
- Top reasons (for example, Delayed Flight) and association rules.
- Word cloud illustration.

## Slide 7 - Predictive Model
- Candidate algorithms and evaluation.
- Selected Linear SVM with 0.76 accuracy.
- Confusion matrix discussion.

## Slide 8 - Web Dashboard Demo
- Walkthrough: prediction form, analytics dashboard, admin review queue.
- Mention database logging and review workflow.

## Slide 9 - Testing & Validation
- Puppeteer automated tests overview plus results summary.
- Future testing recommendations.

## Slide 10 - Conclusion & Next Steps
- Key findings, business implications, and roadmap for future enhancements.
