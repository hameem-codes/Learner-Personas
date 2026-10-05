# Experiment Specifications

## Important Statistical Honesty
The current dataset is observational. We cannot claim that any proposed intervention WILL improve retention, engagement, or learning. The following represent testable product hypotheses based on observed behavioral clusters.

---

# Experiment 1: Weekly Progress Goals
## Target Persona
Weekend Warriors (26.2% of learners)

## Behavioral Evidence
Extremely high `share_sessions_weekend`. These learners naturally batch their activity rather than logging in daily.

## Hypothesis
A weekly progress goal may better align with their observed usage pattern than a strict daily goal, reducing churn caused by daily streak loss.

## Control
Existing daily progress/streak framing UI.

## Treatment
Weekly progress goal UI (e.g., "Complete 5 lessons this week").

## Primary Metric
Weekly active learning days (days per week a user completes at least one lesson).

## Secondary Metrics
Lessons completed per week.

## Guardrail Metrics
7-day retention (must not decrease), overall weekend login rate.

## Expected Direction
Increase in lessons completed per week; stabilization of retention due to less demoralization from broken streaks.

## Risks
Moving away from a daily habit builder might cause some borderline users to forget to log in entirely.

## Success Criteria
Statistically significant positive lift in 7-day retention and weekly lessons completed without degrading daily active user (DAU) baselines drastically.

## Interpretation
If successful, the weekly progress paradigm supports natural batch-learners.

## Follow-up Actions
Roll out flexible scheduling settings globally.

---

# Experiment 2: Rapid Refresher Push
## Target Persona
Dormant Learners (37.1% of learners)

## Behavioral Evidence
Zero median `days_active_last_30` and elevated `days_since_last_active`.

## Hypothesis
Low-friction return experiences (1-minute 'rapid refresher' content) could safely re-engage dormant users overwhelmed by standard lessons.

## Control
Standard re-engagement email ("We miss you! Come back and learn.")

## Treatment
Push notification offering a highly simplified, 1-tap "1-minute refresher" micro-lesson.

## Primary Metric
7-day reactivation rate (percentage of targeted inactive users who return and complete at least one lesson within 7 days of the intervention).

## Secondary Metrics
30-day retention post-reactivation.

## Guardrail Metrics
Notification opt-out rate or app uninstalls.

## Expected Direction
Higher reactivation rate compared to standard emails.

## Risks
Aggressive push notifications may spike uninstalls.

## Success Criteria
Significant lift in reactivation rate without a significant spike in notification opt-outs.

## Interpretation
Proves that high perceived friction blocks dormant users from returning.

## Follow-up Actions
Integrate 'refresher' lessons natively into the onboarding flow for returning users.
