# User Guide - Continuous Learning System

## For Regular Users

### Making Predictions
1. Go to homepage
2. Enter tweet text (10-500 characters)
3. Optionally select airline
4. Click "Analyze Sentiment"
5. Review prediction, confidence, and reasoning

### Understanding Results
- **Sentiment**: negative, neutral, or positive
- **Confidence**: 0-100% (higher = more certain)
- **Key Words**: Most influential words for this prediction
- **Reasoning**: Brief explanation of classification

## For Administrators

### Reviewing Submissions
1. Login to admin panel (password from .env)
2. Click "Pending" tab to see unreviewed predictions
3. For each submission:
   - Select true sentiment from dropdown
   - Optionally add comment
   - Click "Approve" or "Reject"

### Training the Model
1. Ensure 50+ approved reviews with true_sentiment
2. Click "Retrain Model" button
3. Wait for training to complete (~10-30 seconds)
4. Review accuracy comparison modal:
   - Old model accuracy
   - New model accuracy
   - Percentage change
5. Choose "Use New Model" or "Keep Current Model"

### Best Practices
- Always set true_sentiment for approved reviews
- Add comments explaining complex cases
- Retrain after every 50-100 new approvals
- Monitor match/mismatch rate in Approved tab
- Keep old model if new accuracy is lower
