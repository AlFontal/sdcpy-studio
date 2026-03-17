# Understanding SDC and SDC Map

This guide explains the two analysis modes available in `sdcpy-studio`:

- `2-Way SDC`, for studying how two time series move together through time
- `SDC Map`, for studying how a scalar driver is expressed across a gridded field

The goal is not to give implementation notes. The goal is to help you understand what the app is doing, what each parameter changes, and how to read the outputs with confidence.

## Why this analysis exists

Many environmental, climatic, ecological, and health relationships are not stable over a full record.

A single correlation over the whole series can miss the real structure because:

- the relationship may only appear during some periods
- the response may be delayed
- positive and negative events may behave differently
- the coupling may depend on the time scale being examined

Scale-Dependent Correlation, or `SDC`, was designed for that kind of problem. Instead of asking only whether two variables are correlated overall, it asks:

- when are they coupled?
- at what time scale?
- with what lag?
- and, in the mapped workflow, where is the response strongest?

## The two workflows in the app

### 2-Way SDC

Use this when you have two time series and want to explore their local coupling through time.

Typical questions:

- Do these variables move together only during some intervals?
- Does one tend to lead or lag the other?
- Is the relationship stronger at short or long windows?

### SDC Map

Use this when you have:

- one scalar driver series, such as an index or regional summary
- one gridded field, such as temperature, precipitation, pressure, or another spatial variable

Typical questions:

- Where does the field respond to strong positive events in the driver?
- Where does it respond to strong negative events?
- Does the response happen before, near, or after the driver peak?
- Is the lag structure spatially organized?

## How 2-Way SDC works

The `2-Way SDC` workflow compares two series using moving windows.

### Step 1. Choose a window length

The window length defines the temporal scale of the analysis.

Short windows emphasize brief, transient relationships.
Longer windows emphasize more persistent organization.

### Step 2. Slide the window through time

For each valid position, the app extracts a short segment from each series and computes their local correlation.

This means the result is not one number for the whole record. It is a time-localized view of where the two series are more strongly or weakly related.

### Step 3. Repeat across lags

If lag scanning is enabled, the same local comparison is repeated with one series shifted forward or backward in time.

That helps answer whether the relationship is delayed, and whether the delay stays stable or changes through the record.

### What you learn from the result

The result helps you distinguish:

- persistent vs intermittent coupling
- short-lived bursts vs long-lived structure
- lead-lag behavior
- periods where the relationship changes sign

## How SDC Map works

`SDC Map` keeps the same local-correlation logic, but applies it to a scalar driver and a spatial field.

The current workflow in the app is event-conditioned. That means the map is built around selected driver events instead of treating all dates equally.

### Step 1. Align the driver and the field in time

The app first keeps only the dates that are compatible between the scalar driver and the field data.

All later steps work on this aligned time axis.

### Step 2. Detect positive and negative events separately

The driver is searched for local extrema.

The workflow then selects:

- the strongest `N+` positive events
- the strongest `N-` negative events

These are handled separately because positive and negative phases often have different spatial expressions.

### Step 3. Define a base state

The method also needs a reference state that represents relatively quiet conditions.

That base state is defined from the driver using the `beta` threshold. In simple terms:

- strong events are kept as events
- moderate or transitional periods are excluded from the baseline
- the quiet remainder is used as the base state

The field is then interpreted relative to that background state.

### Step 4. Build event-centered windows

For each selected event, the app builds a centered window with width `r_w`, shown in the interface as `Correlation width`.

This window defines the local temporal neighborhood used for the analysis.

### Step 5. Scan field responses across lags

At each grid cell, the app compares the local driver event window with local field windows across the lag range you requested.

This produces an event-local, lag-aware response estimate for each cell.

### Step 6. Keep statistically supported responses

The local correlations are filtered using the selected significance settings and permutation testing.

This reduces the number of weak or unstable responses that would otherwise clutter the map.

### Step 7. Average within each event class

The selected positive events are summarized together, and the selected negative events are summarized together.

That is why the app produces separate positive and negative outputs.

## What the event preview means

Before you run a map, the app shows a preview of the detected events on the driver series.

This preview helps you check whether the selected `N+`, `N-`, `beta`, and `r_w` values are sensible before you launch a more expensive run.

The preview shows:

- the driver series
- selected positive peaks
- selected negative troughs
- the base-state threshold
- the event windows implied by `r_w`

Use it as a quality-control step. If the selected events do not match your scientific question, adjust the parameters before running the map.

## What the main SDC Map parameters mean

### Positive events (`N+`)

How many positive driver events are included in the positive-class analysis.

### Negative events (`N-`)

How many negative driver events are included in the negative-class analysis.

### Base-state beta

Controls how strict the baseline definition is.

Smaller values define a narrower quiet baseline.
Larger values allow more dates into the base state.

### Correlation width (`r_w`)

The width of each event-centered analysis window.

Shorter widths focus on local, sharper responses.
Longer widths allow broader event evolution to contribute.

### Lag range

Defines how far before and after the driver event the field response is searched.

### Significance alpha and permutations

Control the statistical filtering applied to local responses.

More permutations generally provide a more stable null estimate but take longer.

## Reading the SDC Map outputs

The app gives you two complementary views:

- a static `A/B/C/D` summary
- a dynamic lag explorer

### Static map A. Correlation

This map shows the average maximum supported SDC correlation at each grid cell.

It answers:

> Where is the local relationship strongest?

### Static map B. Position

This map shows where, within the event-centered window, the strongest field response tends to occur relative to the driver event peak.

It answers:

> Does the field respond earlier, near the peak, or later within the event window?

### Static map C. Lag

This map shows the lag at which the strongest supported response is found.

It answers:

> How many time steps earlier or later does the response appear?

### Static map D. Timing

This map combines the event-relative position and lag information into a single timing summary.

It answers:

> What is the overall timing of the response with respect to the driver event?

### Dynamic lag explorer

The lag explorer shows the mapped response at one selected lag at a time.

Use it when you want to see how the response pattern changes through lag space instead of looking only at the condensed summary layers.

## Positive and negative maps should not be assumed to match

One of the main reasons the workflow separates `N+` and `N-` is that many systems are asymmetric.

That means:

- positive events may produce strong, coherent maps
- negative events may be weaker or shifted
- the timing structure may differ between signs

This is expected and can be scientifically informative.

## What custom uploads are for

The app supports:

- custom `CSV` driver uploads
- custom `NetCDF` field uploads

This lets you use the same event-conditioned map workflow on your own data rather than only on the bundled catalog datasets.

The inspection step checks the uploaded files before the run:

- for drivers, it looks for a usable date column and numeric series
- for fields, it checks time, latitude, longitude, and any extra selectable dimensions such as level

## Practical advice before running a map

### Start simple

Use modest values for:

- `N+`
- `N-`
- lag range
- permutations

Then increase complexity once the event selection and spatial behavior look sensible.

### Check the event preview first

If the chosen peaks do not look right, the final map will not answer the question you care about.

### Do not over-read empty maps

Sometimes a run completes but yields very few valid cells. That can happen if:

- the chosen parameters are too strict
- the selected events do not produce a stable response
- the field or time window is not suitable for the question

### Use the class split as information

If the positive class is structured and the negative class is weak, that is already a result. It does not automatically mean the run failed.

## What the method is best at

This workflow is most useful when you expect:

- intermittent coupling
- event-driven responses
- lagged teleconnections
- sign asymmetry
- scale dependence

It is less useful if your question can already be answered with one stable full-series correlation.

## What the app does not claim

The app helps you localize and summarize structured relationships. It does not by itself prove causality.

The outputs should be interpreted together with:

- domain knowledge
- data quality checks
- sensitivity analysis across parameter choices
- and, where possible, external validation

## A sensible user workflow

For most studies, a good order is:

1. Explore the driver and target behavior in `2-Way SDC`.
2. Move to `SDC Map` once you have a plausible event-based question.
3. Check the event preview carefully.
4. Run modest parameter settings first.
5. Inspect both the static summary maps and the lag explorer.
6. Compare positive and negative classes explicitly.
7. Export the outputs you want to report.

## In short

`2-Way SDC` tells you when two series are locally coupled.

`SDC Map` tells you where and when a field responds to selected driver events, with positive and negative phases treated separately and the response summarized both statically and by lag.
