## Project Context

### Project Overview
The Data Chat agent monitors Slack conversations, captures user questions about available datasets, and returns concise, data-backed answers. It enriches user understanding by extracting and synthesizing insights from one or more CSV files so teams can quickly act on the latest metrics.

### System Scope
- In scope: Slack question intake, question parsing and routing, semantic access to CSV-backed data, automated insight generation, formatted responses for Slack.
- Out of scope: CSV creation or maintenance, non-Slack interfaces, persistent data storage beyond source CSVs, long-term analytics pipelines.

### Architecture Summary
- Intake subsystem captures Slack questions, interprets intent, selects relevant CSV sources, and exposes a semantic layer for structured queries.
- Engine subsystem executes analytics logic, combining data from the selected CSV files to derive actionable insights.
- Output subsystem converts analytic results into clear Slack-ready answers aligned with the original question.

### Key Inputs and Outputs
- Inputs: User-authored Slack questions, one or more CSV files containing the underlying data.
- Outputs: Structured Slack responses summarizing findings, with any supporting metrics or observations surfaced from the CSV data.

### Design Rationale
- The intake engine keeps the solution flexible by mapping natural language questions to the correct datasets, minimizing manual routing.
- A dedicated insight engine supports multi-CSV analysis, ensuring responses combine all relevant data without extra user effort.
- The output stage generates Slack-native answers, meeting users where they work and accelerating decisions without requiring dashboard navigation.