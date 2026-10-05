# Learner Personas

## Methodology
- **Dataset**: 6,000 learners.
- **Segmentation**: 4 personas derived from K-Means on behavior_robust data.
- **Evidence-Based Interpretation**: All persona traits are bounded strictly to observed median/mean differences. No causal or psychological inferences are made.

## Persona 1: Dormant Learners (37.1%)
**Behavioral signature:** Users exhibiting minimal recent engagement and elevated periods of inactivity.
- **Key metrics**: Extremely low `days_active_last_30` and `sessions_per_week`; elevated `days_since_last_active`.
- **Strongest differentiators**: `days_since_last_active` is substantially higher than the population average.
- **Potential needs**: Lower-friction content or re-engagement pathways.
- **Potential product opportunities**: 
  *Hypothesis:* Push notifications with one-tap, highly simplified "refresher" content could re-engage users.
  *Experiment:* A/B test a standard re-engagement email vs. an in-app "1-minute refresher" prompt.
- **Evidence**: Median days active last 30 is 0. 
- **Limitations**: The dataset does not capture the reason for inactivity (e.g., technical issues, alternative resources, or lack of time).

## Persona 2: Streak Maintainers (27.3%)
**Behavioral signature:** Users demonstrating high login frequency and long consecutive active days, coupled with shorter average session durations.
- **Key metrics**: High `longest_streak_days` and `days_active_last_30`; lower `avg_session_minutes`.
- **Strongest differentiators**: Massive standardized effect size on `longest_streak_days`.
- **Potential needs**: Ability to maintain consistent engagement even on low-capacity days.
- **Potential product opportunities**: 
  *Hypothesis:* Micro-lessons (1-2 minutes) may support consistent engagement on busy days.
  *Experiment:* A/B test introducing 1-minute "rapid review" modules to measure impact on streak retention.
- **Evidence**: Exhibits the highest `days_active_last_30` alongside shortest `avg_session_minutes`.
- **Limitations**: High engagement metrics do not inherently prove high knowledge retention or course mastery.

## Persona 3: Weekend Warriors (26.2%)
**Behavioral signature:** Users who concentrate a large majority of their learning activity on weekends.
- **Key metrics**: Exceptionally high `share_sessions_weekend`; lower overall `sessions_per_week` but longer `avg_session_minutes`.
- **Strongest differentiators**: `share_sessions_weekend` is the defining feature.
- **Potential needs**: Flexibility to learn in concentrated batches rather than strictly daily requirements.
- **Potential product opportunities**: 
  *Hypothesis:* Weekly progress goals may fit this behavior better than daily streaks.
  *Experiment:* A/B test a weekly progress bar UI vs. a daily streak UI for this segment to measure overall retention.
- **Evidence**: `share_sessions_weekend` is heavily elevated compared to the population mean.
- **Limitations**: We cannot deduce demographics (e.g., student vs. working professional) from the weekend behavior alone.

## Persona 4: Power Learners (9.4%)
**Behavioral signature:** Users exhibiting massive learning volume, exceptionally high XP accumulation, and active leaderboard participation.
- **Key metrics**: Exceptionally high `xp_earned_90d`, `lessons_completed_90d`, and `leaderboard_weeks_joined`.
- **Strongest differentiators**: XP and lessons completed show massive effect sizes over the baseline.
- **Potential needs**: Advanced content, continuous challenges, and social/competitive outlets.
- **Potential product opportunities**: 
  *Hypothesis:* Elite leaderboard tiers or advanced/difficult course tracks might increase long-term retention for extremely active users.
  *Experiment:* A/B test unlocking a "Diamond League" or providing beta access to harder content.
- **Evidence**: Top decile across all volume and intensity metrics.
- **Limitations**: As an outlier segment, their behavior patterns do not generalize to the broader population.
