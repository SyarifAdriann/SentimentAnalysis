# Sentiment Pattern Analysis

## Sentiment by Airline
| airline        |   negative |   neutral |   positive |   total |   negative_pct |   neutral_pct |   positive_pct |
|:---------------|-----------:|----------:|-----------:|--------:|---------------:|--------------:|---------------:|
| American       |       1960 |       463 |        336 |    2759 |          71.04 |         16.78 |          12.18 |
| Delta          |        955 |       723 |        544 |    2222 |          42.98 |         32.54 |          24.48 |
| Southwest      |       1186 |       664 |        570 |    2420 |          49.01 |         27.44 |          23.55 |
| US Airways     |       2263 |       381 |        269 |    2913 |          77.69 |         13.08 |           9.23 |
| United         |       2633 |       697 |        492 |    3822 |          68.89 |         18.24 |          12.87 |
| Virgin America |        181 |       171 |        152 |     504 |          35.91 |         33.93 |          30.16 |

## Negative Reasons by Airline (Top 20)
| airline    | negativereason              |   count |
|:-----------|:----------------------------|--------:|
| US Airways | Customer Service Issue      |     811 |
| American   | Customer Service Issue      |     768 |
| United     | Customer Service Issue      |     681 |
| United     | Late Flight                 |     525 |
| US Airways | Late Flight                 |     453 |
| Southwest  | Customer Service Issue      |     391 |
| United     | Can't Tell                  |     379 |
| United     | Lost Luggage                |     269 |
| Delta      | Late Flight                 |     269 |
| American   | Late Flight                 |     249 |
| American   | Cancelled Flight            |     246 |
| US Airways | Can't Tell                  |     246 |
| United     | Bad Flight                  |     216 |
| Delta      | Customer Service Issue      |     199 |
| American   | Can't Tell                  |     198 |
| US Airways | Cancelled Flight            |     189 |
| Delta      | Can't Tell                  |     186 |
| United     | Cancelled Flight            |     181 |
| United     | Flight Attendant Complaints |     168 |
| Southwest  | Cancelled Flight            |     162 |

## Association Rules
| antecedents                                     | consequents        |   support |   confidence |   lift |   leverage |   conviction |
|:------------------------------------------------|:-------------------|----------:|-------------:|-------:|-----------:|-------------:|
| airline=American                                | sentiment=negative | 0.213554  |            1 |      1 |          0 |          inf |
| airline=Southwest                               | sentiment=negative | 0.129222  |            1 |      1 |          0 |          inf |
| airline=Delta                                   | sentiment=negative | 0.104053  |            1 |      1 |          0 |          inf |
| airline=United                                  | sentiment=negative | 0.286882  |            1 |      1 |          0 |          inf |
| airline=US Airways                              | sentiment=negative | 0.246568  |            1 |      1 |          0 |          inf |
| airline=United, reason=Flight Booking Problems  | sentiment=negative | 0.0156897 |            1 |      1 |          0 |          inf |
| airline=United, reason=Lost Luggage             | sentiment=negative | 0.0293092 |            1 |      1 |          0 |          inf |
| reason=Flight Attendant Complaints              | sentiment=negative | 0.0524079 |            1 |      1 |          0 |          inf |
| reason=Customer Service Issue                   | sentiment=negative | 0.317063  |            1 |      1 |          0 |          inf |
| reason=Cancelled Flight                         | sentiment=negative | 0.0922859 |            1 |      1 |          0 |          inf |
| reason=Can't Tell                               | sentiment=negative | 0.129658  |            1 |      1 |          0 |          inf |
| reason=Bad Flight                               | sentiment=negative | 0.0631946 |            1 |      1 |          0 |          inf |
| airline=Virgin America                          | sentiment=negative | 0.0197211 |            1 |      1 |          0 |          inf |
| reason=Flight Booking Problems                  | sentiment=negative | 0.0576378 |            1 |      1 |          0 |          inf |
| airline=American, reason=Late Flight            | sentiment=negative | 0.0271301 |            1 |      1 |          0 |          inf |
| airline=American, reason=Customer Service Issue | sentiment=negative | 0.0836784 |            1 |      1 |          0 |          inf |
| airline=American, reason=Cancelled Flight       | sentiment=negative | 0.0268032 |            1 |      1 |          0 |          inf |
| airline=American, reason=Can't Tell             | sentiment=negative | 0.0215733 |            1 |      1 |          0 |          inf |
| reason=longlines                                | sentiment=negative | 0.0193942 |            1 |      1 |          0 |          inf |
| reason=Lost Luggage                             | sentiment=negative | 0.0788843 |            1 |      1 |          0 |          inf |

## Word Cloud
![Negative sentiment word cloud](C:/xampp/htdocs/SentimentAnalysis/visualizations/wordcloud_negative.png)