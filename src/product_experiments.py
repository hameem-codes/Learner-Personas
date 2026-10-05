import os
import pandas as pd
from pathlib import Path

from src.config import DATA_DIR, PROJECT_ROOT

PRODUCT_DIR = DATA_DIR / 'outputs' / 'product'
REPORTS_PRODUCT_DIR = PROJECT_ROOT / 'reports' / 'product'

PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

def generate_product_hypotheses():
    hypotheses = [
        {
            "persona": "Dormant Learners",
            "cluster_id": 1,
            "observed_behavior": "Zero median days active last 30 and elevated days since last active.",
            "evidence_feature": "days_active_last_30, days_since_last_active",
            "hypothesized_need": "Low-friction return experience to restart learning without overwhelm.",
            "proposed_intervention": "1-minute 'rapid refresher' content push notifications.",
            "assumption": "Users churned due to high perceived effort of full lessons.",
            "success_metric": "7-day reactivation rate (percentage of inactive users returning and completing 1+ lessons).",
            "guardrail_metric": "Notification opt-out rate.",
            "experiment_type": "A/B test (Targeted email vs Push notification)."
        },
        {
            "persona": "Dormant Learners",
            "cluster_id": 1,
            "observed_behavior": "Low session duration and frequency before churning.",
            "evidence_feature": "sessions_per_week, avg_session_minutes",
            "hypothesized_need": "Content rediscovery aligned to current goals.",
            "proposed_intervention": "Goal recalibration quiz on next login.",
            "assumption": "Users lost sight of their learning objectives.",
            "success_metric": "30-day retention post-login.",
            "guardrail_metric": "Quiz drop-off rate.",
            "experiment_type": "A/B test (Standard dashboard vs recalibration popup)."
        },
        {
            "persona": "Streak Maintainers",
            "cluster_id": 0,
            "observed_behavior": "Extremely high longest streak but low average session minutes.",
            "evidence_feature": "longest_streak_days, avg_session_minutes",
            "hypothesized_need": "Short-session learning to preserve streaks on busy days.",
            "proposed_intervention": "1-2 minute micro-lessons.",
            "assumption": "Users log in just to keep streak but lack time for full lessons.",
            "success_metric": "Streak continuation rate (proportion maintaining streak past 30 days).",
            "guardrail_metric": "Total learning time per week (to ensure cannibalization of long sessions doesn't occur).",
            "experiment_type": "A/B test (Micro-lesson availability)."
        },
        {
            "persona": "Weekend Warriors",
            "cluster_id": 3,
            "observed_behavior": "Activity heavily concentrated on weekends.",
            "evidence_feature": "share_sessions_weekend",
            "hypothesized_need": "Flexible scheduling that doesn't penalize weekday inactivity.",
            "proposed_intervention": "Weekly progress goals instead of daily streaks.",
            "assumption": "Daily streak pressure causes demotivation for natural batch learners.",
            "success_metric": "Lessons completed per week.",
            "guardrail_metric": "Weekend login rate (should not drop).",
            "experiment_type": "A/B test (Daily streak UI vs Weekly progress bar)."
        },
        {
            "persona": "Power Learners",
            "cluster_id": 2,
            "observed_behavior": "Exceptionally high XP earned and lessons completed.",
            "evidence_feature": "xp_earned_90d, lessons_completed_90d",
            "hypothesized_need": "Advanced content and deeper progression systems.",
            "proposed_intervention": "Elite leaderboard tiers (e.g., Diamond League) and challenge tracks.",
            "assumption": "Power users risk churning if they 'beat' the current system and get bored.",
            "success_metric": "90-day retention rate of power users.",
            "guardrail_metric": "Content completion quality (e.g. quiz scores).",
            "experiment_type": "A/B test (Standard leaderboard vs Elite tiers)."
        }
    ]
    
    df = pd.DataFrame(hypotheses)
    df.to_csv(PRODUCT_DIR / 'product_hypotheses.csv', index=False)
    return df

def generate_experiment_prioritization():
    priorities = [
        {
            "experiment_name": "Weekly Progress Goals (Weekend Warriors)",
            "potential_impact": "High",
            "evidence_strength": "High",
            "implementation_complexity": "Medium",
            "population_size": "26.2%",
            "testability": "High",
            "priority_score": 1
        },
        {
            "experiment_name": "Rapid Refresher Push (Dormant Learners)",
            "potential_impact": "Medium",
            "evidence_strength": "High",
            "implementation_complexity": "Low",
            "population_size": "37.1%",
            "testability": "High",
            "priority_score": 2
        },
        {
            "experiment_name": "Micro-lessons (Streak Maintainers)",
            "potential_impact": "Medium",
            "evidence_strength": "High",
            "implementation_complexity": "High",
            "population_size": "27.3%",
            "testability": "Medium",
            "priority_score": 3
        },
        {
            "experiment_name": "Elite Tiers (Power Learners)",
            "potential_impact": "Medium",
            "evidence_strength": "Medium",
            "implementation_complexity": "Medium",
            "population_size": "9.4%",
            "testability": "High",
            "priority_score": 4
        }
    ]
    
    df = pd.DataFrame(priorities)
    df.to_csv(PRODUCT_DIR / 'experiment_prioritization.csv', index=False)

def generate_persona_product_matrix(hypotheses_df):
    matrix_data = []
    
    for persona in hypotheses_df['persona'].unique():
        sub = hypotheses_df[hypotheses_df['persona'] == persona].iloc[0]
        matrix_data.append({
            "persona": persona,
            "observed behavior": sub["observed_behavior"],
            "potential need": sub["hypothesized_need"],
            "recommended intervention": sub["proposed_intervention"],
            "primary experiment": sub["experiment_type"],
            "primary metric": sub["success_metric"],
            "guardrail": sub["guardrail_metric"]
        })
        
    df = pd.DataFrame(matrix_data)
    df.to_csv(PRODUCT_DIR / 'persona_product_matrix.csv', index=False)

def generate_experiment_specs():
    content = """# Experiment Specifications

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
"""
    with open(REPORTS_PRODUCT_DIR / 'experiment_specs.md', 'w') as f:
        f.write(content)

def main():
    df = generate_product_hypotheses()
    generate_experiment_prioritization()
    generate_persona_product_matrix(df)
    generate_experiment_specs()
    print("Product hypothesis and experiment design generated successfully.")

if __name__ == '__main__':
    main()
