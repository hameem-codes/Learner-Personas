# Learner Personas

## Methodology
- **Dataset**: Genuine 6,000 learner personas dataset from Data Career School.
- **Preprocessing**: Robust scaling applied to handle extreme behavioral outliers.
- **Clustering**: Evaluated K=2 to K=10 using K-Means and Agglomerative clustering.
- **Why K=4**: K=2 strictly divided active vs inactive users. K=4 cleanly extracted three highly distinct active sub-populations (Streak, Weekend, Power) with near-perfect stability (ARI=0.9998) without creating micro-clusters.
- **Why RobustScaler**: Successfully mitigated the influence of the 230 extreme power users, keeping them as a cohesive 9.4% segment rather than splitting them into useless singleton clusters.
- **Validation**: Strict deterministic assertions and manual audit confirmed robust separation.

## Persona 1: Dormant Learners (Cluster 1, 37.1%)
**ONE-LINE DESCRIPTION:** Users who have functionally churned and exhibit near-zero recent activity.
- **Learning pattern**: Negligible XP and lessons completed in the last 90 days.
- **Engagement pattern**: Zero or near-zero active days in the last 30 days.
- **Retention pattern**: High days since last active; broken streaks.
- **Typical habits**: None measurable currently.
- **Key differentiators**: `days_since_last_active` is exceptionally HIGH; all engagement is LOW.
- **Potential needs**: A low-friction reason to return.
- **Potential friction points**: Content might have been too hard, or life got busy.
- **Potential product opportunities**: Win-back email campaigns highlighting new, easy "welcome back" content.
- **Evidence**: Median days active last 30 = 0.
- **Limitations**: We do not know *why* they left.

## Persona 2: Streak Maintainers (Cluster 0, 27.3%)
**ONE-LINE DESCRIPTION:** Highly consistent daily users who complete quick sessions to keep their streak alive.
- **Learning pattern**: Moderate XP and lessons.
- **Engagement pattern**: Very high days active, but shorter average session minutes.
- **Retention pattern**: Massive longest streaks.
- **Typical habits**: Daily logins, likely mobile-first.
- **Key differentiators**: `longest_streak_days` and `days_active_last_30` are HIGH.
- **Potential needs**: Efficiency and streak protection.
- **Potential friction points**: Long, unskippable lessons might break their habit.
- **Potential product opportunities**: Streak freezes, micro-lessons (1-2 minutes).
- **Evidence**: Highest median streak days among all groups.
- **Limitations**: May not be actually acquiring deep knowledge, just gaming the system.

## Persona 3: Weekend Warriors (Cluster 3, 26.2%)
**ONE-LINE DESCRIPTION:** Casual learners who concentrate their activity into fewer, longer sessions, mostly on weekends.
- **Learning pattern**: Normal XP, but achieved in fewer days.
- **Engagement pattern**: Low sessions per week, but longer session minutes.
- **Retention pattern**: Moderate streaks, often broken mid-week.
- **Typical habits**: High `share_sessions_weekend`.
- **Key differentiators**: `share_sessions_weekend` is extremely HIGH.
- **Potential needs**: Flexibility to skip weekdays without penalty.
- **Potential friction points**: Daily streak mechanics are highly demotivating for them.
- **Potential product opportunities**: "Weekend Quest" challenges; weekly goals instead of daily streaks.
- **Evidence**: `share_sessions_weekend` median is highly elevated compared to the population.
- **Limitations**: We can't distinguish between "busy professionals" and "casual hobbyists".

## Persona 4: Power Learners (Cluster 2, 9.4%)
**ONE-LINE DESCRIPTION:** Voracious learners with extremely high engagement, XP, and lesson completion.
- **Learning pattern**: Massive XP and completed lessons.
- **Engagement pattern**: High sessions per week, high session minutes.
- **Retention pattern**: Long streaks, deeply retained.
- **Typical habits**: Highly active on leaderboards.
- **Key differentiators**: `xp_earned_90d`, `lessons_completed_90d`, `leaderboard_weeks_joined` are exceptionally HIGH.
- **Potential needs**: Recognition, advanced content, social status.
- **Potential friction points**: Running out of fresh content; lack of challenge.
- **Potential product opportunities**: Elite tiers/leaderboards, advanced/difficult course tracks, beta-testing access.
- **Evidence**: Top 10% of XP distribution; heavily skewed behavioral metrics.
- **Limitations**: Outlier status means standard retention metrics don't apply.

## Persona Comparison
| Dimension | Dormant Learners | Streak Maintainers | Weekend Warriors | Power Learners |
|---|---|---|---|---|
| Engagement | LOW | HIGH | LOW-MED | VERY HIGH |
| Learning Intensity | LOW | MED | MED | VERY HIGH |
| Retention | LOW | HIGH | MED | HIGH |
| Session Duration | N/A | LOW | HIGH | HIGH |
| Streak Behavior | N/A | HIGH | LOW | HIGH |
| Content Exploration| LOW | MED | MED | HIGH |
| Weekend Behavior | N/A | NORMAL | HIGH | NORMAL |

## Product Opportunities (Hypotheses)
1. **Weekend Warrior Goals**: Introduce weekly progress bars instead of purely daily streaks, as daily streaks heavily penalize the 26.2% of users who prefer weekend batching.
2. **Micro-Lessons for Streak Maintainers**: Provide 1-minute "review" exercises for Streak Maintainers to easily secure their daily habit on busy days.
3. **Elite Leaderboard Tiers**: Create an exclusive league for Power Learners to prevent boredom and retain the top 9.4% engagement drivers.
4. **Win-Back Easy Starts**: Send targeted push notifications to Dormant Learners offering a one-tap, no-fail "refresher" lesson.
5. **Content Discovery**: Recommend new courses heavily to Power Learners who rapidly consume existing content and have a high share of lessons on new content.

## Limitations
- We cannot infer causality (e.g., do leaderboards cause Power Learners to study more, or do Power Learners naturally join leaderboards?).
- No demographic data exists to contextualize *why* Weekend Warriors only study on weekends (e.g., student vs. employed).
