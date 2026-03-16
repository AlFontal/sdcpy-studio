# SDC and SDC-map: detailed methodological explanation

## Source and scope

This note was reconstructed from the slide deck **“SDC: A transient-analysis tool” by Xavier Rodó** and related slides on multiscale and spatiotemporal applications. It explains, in technical but readable terms, what the **Scale-Dependent Correlation (SDC)** algorithm does and how the **SDC-map** extends that logic to spatiotemporal fields.

Where the slides are schematic rather than fully formal, I make the inference explicit and keep the interpretation conservative.

---

## 1. Why SDC exists

The main motivation behind SDC is that many climate, environmental, and epidemiological relationships are **not stationary**, **not continuous in time**, and often **not well captured by one global correlation coefficient**.

A standard Pearson correlation over a full time series can be misleading when:

- the relationship is present only during certain intervals,
- the forcing is intermittent or threshold-like,
- the response depends on the background state of the impacted system,
- the lag between driver and response changes over time,
- the signal is visible at some temporal scales but not at others.

The slide deck states this very directly: climate forcing may be **discontinuous**, may act only under selected situations, and may only become visible when local variability conditions favor interaction. In other words, the problem is not just “is X correlated with Y?”, but rather:

- **when** are X and Y coupled?
- **at what scale** is the coupling visible?
- **with what lag** does the coupling appear?
- **where** in space is the response expressed?
- **during which types of events** does the coupling emerge?

SDC was designed to answer exactly that class of questions.

---

## 2. Conceptual summary in one paragraph

**SDC is a local, scale-aware correlation framework**. Instead of correlating two full series once, it computes correlations on **moving windows of a chosen size** and, optionally, at **different time lags**. This reveals whether two signals are coupled only in some intervals, whether the coupling changes sign, and whether it depends on the timescale considered. The **SDC-map** extends this idea by taking one scalar time series, selecting its most relevant positive and negative events, and computing **lagged local correlation maps** against a gridded field, after filtering out a “base state”. The final product is a spatial map of where and when the field responds to the selected events.

---

## 3. What the basic SDC algorithm does

## 3.1 Core idea

Given two time series, say:

- `x(t)`: a driver or candidate forcing series,
- `y(t)`: a response or target series,

SDC does not compute one single correlation for the whole record. Instead, it computes correlations between **subsegments** of the series.

That means the algorithm asks:

> If I only look at a segment of length `s`, are the two signals correlated there?

Then it slides that segment across time, repeating the calculation. This produces a matrix or set of local correlation values.

This is why the method is called **scale-dependent**:

- the **scale** is the segment/window length `s`,
- the result depends on that scale,
- the analyst is expected to inspect several values of `s`.

---

## 3.2 The problem it is trying to solve

The deck gives an illustrative case: two series can have a clear transient interval of strong coupling, but the full-series correlation can still be low, around `0.3` to `0.4`. In such cases, the global coefficient hides the structure.

SDC recovers that structure by localizing the correlation in time.

So the goal is not just sensitivity, but **localization**:
- localization in time,
- localization in lag,
- and, in the mapped version, localization in space.

---

## 3.3 Typical temporal SDC workflow

A practical temporal SDC analysis usually works like this:

### Step 1. Choose a window length `s`
This is the analysis scale, also described in the slides as the segment length.

Examples:
- short windows for transient, rapid interactions,
- longer windows for more persistent, lower-frequency coupling.

The slides emphasize that one should usually inspect **several segment lengths**, because different structures become visible at different scales.

### Step 2. Extract paired local segments
At each valid time index, the algorithm takes:
- a window of length `s` from `x`,
- a corresponding window of length `s` from `y`.

If a lag is being considered, one of those windows is shifted relative to the other.

### Step 3. Compute the local correlation
For each pair of local windows, compute a correlation coefficient, usually Pearson correlation.

This produces a local measure of similarity:
- positive if both vary together,
- negative if one rises while the other falls,
- near zero if no linear coupling is visible on that segment.

### Step 4. Repeat through time
Slide the window through the record and repeat.

### Step 5. Optionally repeat across lags
To test delayed responses, repeat the calculation for different lags.

### Step 6. Inspect the local correlation pattern
The output is a time-localized, scale-specific representation of coupling, often visualized as an SDC plot.

---

## 4. What an SDC plot shows

Although the slides are graphical, the interpretation is fairly clear.

An SDC plot typically reveals:

- **persistent coupling**: blocks or bands of repeated positive or negative correlation,
- **transient coupling**: isolated patches,
- **intermittent forcing**: repeated but discontinuous patches,
- **different noise structures**: different textures or patterns depending on the autocorrelation structure of the signals,
- **changing dominant periodicity**: the shape of the local correlation field differs depending on the underlying dynamics.

The deck explicitly notes that SDC can distinguish:
- different types of noise,
- different types of couplings,
- and especially temporary interactions and discontinuities.

That is important: SDC is not just descriptive. It is meant as a **pattern recognition tool** for dynamical relationships.

---

## 5. Why scale matters

One of the most important points in the presentation is that the answer depends on the scale `s`.

A relationship can be:
- weak at short scales but clear at long scales,
- absent at long scales but strong at short scales,
- consistent across scales,
- or only present within a narrow scale band.

This is why the method is often run in **multiscale mode**, for example with `s` varying from 5 to 100 days in the respiratory disease applications shown in the deck.

This gives two benefits:

1. It tests robustness.
   If the coupling is real, it may persist across neighboring scales instead of appearing only at one arbitrary window size.

2. It helps distinguish mechanisms.
   Short-scale coupling may reflect immediate triggering or weather events.
   Long-scale coupling may reflect seasonal organization, background state, or slowly evolving modes.

The slides on influenza and COVID-19 applications suggest exactly this logic: associations can be examined both at large scales and at smaller local scales, with attention to whether they reflect real structure or spurious seasonality.

---

## 6. How lags are handled in SDC

SDC is especially useful when the effect of one process on another is delayed.

Rather than assuming a fixed lag over the entire record, the method evaluates local correlation at one or several candidate lags. That means the analyst can see:

- whether the lag changes over time,
- whether stronger correlations tend to happen at short or long lags,
- whether positive and negative phases behave differently.

The deck explicitly remarks in one example that **stronger correlations are associated with shorter lags in general**, but only during stages of high transmission. That is a very characteristic SDC finding: lag structure itself can be state dependent.

---

## 7. What makes SDC different from ordinary rolling correlation

At first glance, SDC may sound like rolling correlation. But the intended use is more structured and more interpretive.

SDC differs in emphasis because it is designed to study:

- **scale dependence** explicitly,
- **lag structure** explicitly,
- **transient couplings** rather than only smoothed tracking,
- **pattern interpretation** across scales and lags,
- and later, **event-conditioned spatial composites** in SDC-map.

So while rolling correlation is one computational building block, SDC is better understood as a **framework for local, multiscale, lagged coupling analysis**.

---

## 8. A formalized version of temporal SDC

The slides do not present explicit equations, but the method can be described in operational terms.

Let:

- `x_t`, `t = 1, ..., T`
- `y_t`, `t = 1, ..., T`

Choose:
- window length `s`,
- lag `ℓ`.

For each admissible time index `τ`, define:

- `X(τ) = [x_τ, x_(τ+1), ..., x_(τ+s-1)]`
- `Y(τ, ℓ) = [y_(τ+ℓ), y_(τ+ℓ+1), ..., y_(τ+ℓ+s-1)]`

Then compute:

- `r(τ, ℓ, s) = corr(X(τ), Y(τ, ℓ))`

This gives a local correlation value for:
- time location `τ`,
- lag `ℓ`,
- scale `s`.

Repeat across all valid `τ`, and possibly across multiple `ℓ` and `s`.

The output is then:
- a vector of local correlations if lag and scale are fixed,
- a matrix if time and lag are both scanned,
- a family of plots if multiple scales are explored.

That is the temporal SDC core.

---

## 9. How the presentation interprets SDC scientifically

The deck uses several examples:

- ENSO versus cholera,
- ENSO versus other climatic indices,
- influenza versus meteorological variables,
- COVID-19 versus flu-like environmental organization,
- drought and large-scale climate forcing.

The scientific logic is consistent across them:
1. global correlation may be modest,
2. forcing may be intermittent,
3. different events may have different expressions,
4. local analysis can reveal which portions of the time series drive the apparent association.

This is especially useful in climate-health systems, where transmission is often threshold-dependent and modulated by host susceptibility, environment, and season.

---

## 10. From SDC to SDC-map

The temporal SDC algorithm analyzes local correlation between two time series. The **SDC-map** extends this idea to a different data configuration:

- one **scalar time series**, for example Niño3.4 index or disease incidence,
- one **spatiotemporal field**, for example sea level pressure, SST, temperature, or another gridded atmospheric/oceanic variable.

The purpose is to determine:

- **where** the field is significantly associated with selected events in the scalar series,
- **at which lags** the association appears,
- and how the response differs between positive and negative events.

This is why the deck calls it the **spatiotemporal approach**.

---

## 11. What SDC-map does in plain language

SDC-map takes the logic of local correlation and makes it event-centered:

1. identify the strongest positive and negative events in a reference scalar series,
2. define a “base state” that represents normal or moderate conditions,
3. remove that base state from the gridded field,
4. compute local lagged correlations between each selected event and the field at each grid point,
5. assess statistical significance,
6. average over the selected events,
7. create maps of the resulting correlation structure.

The result is a spatial picture of where the atmosphere, ocean, climate, or another field responds to selected phases of the scalar driver.

---

## 12. Inputs to SDC-map

The slide deck is very clear about the required inputs.

### 12.1 Scalar time series
A one-dimensional series such as:
- a disease series,
- Niño3.4,
- SOI,
- drought energy,
- or another regional index.

This scalar series is the event-defining series.

### 12.2 Spatiotemporal matrix
A gridded variable indexed by space and time, such as:
- sea level pressure,
- SST,
- 2 m temperature,
- atmospheric fields,
- or another climate variable.

This is the field in which one wants to detect the spatial imprint of the scalar driver.

---

## 13. Parameters in SDC-map

The slides list three main parameters.

### 13.1 `N+` and `N-`
Number of selected positive peaks and negative peaks from the scalar series.

These define how many strong positive and negative events are included in the analysis.

For example:
- the strongest 3 warm events,
- the strongest 3 cold events.

### 13.2 `β` (base-state robustness parameter)
This parameter is used to define how much of the scalar series around the event peaks is considered part of the event class versus the “base state”.

The slides show:
- red area: selected and ignored peaks,
- grey area: moderate years,
- white area: normal years (base state),
- and a threshold involving `β · x0`.

Interpretation:
- `x0` is a reference event magnitude or threshold,
- `β` scales the robustness of the base state,
- the purpose is to isolate a clean non-event baseline.

In practice, `β` controls how strictly one excludes transitional or moderate conditions from the baseline.

### 13.3 `r_w` (correlation width)
The width of the local correlation window.

In the example slide:
- `r_w = 25 months`.

This is the temporal neighborhood used when computing local correlation around an event for a given lag.

---

## 14. Peak selection in SDC-map

The deck devotes multiple slides to peak selection, which shows how central it is.

### 14.1 Why peaks are selected
The method is not interested in all times equally. It assumes that the forcing signal may be strongest during specific positive or negative episodes.

For ENSO-type applications, for example:
- positive peaks may correspond to El Niño years,
- negative peaks to La Niña years.

### 14.2 Positive and negative peaks are treated separately
The algorithm selects:
- the top `N+` positive peaks,
- the top `N-` negative peaks.

This separation matters because the spatial response to positive and negative phases may not be symmetric.

### 14.3 Ignored peaks versus base state
The slides distinguish:
- selected peaks,
- ignored peaks,
- moderate years,
- normal years.

This suggests the following operational logic:

- selected peaks are used to construct event correlations,
- some other strong-ish peaks may be excluded if not part of the chosen set,
- moderate years are not used as baseline,
- only “normal years” form the base state.

This is a very important methodological choice because it tries to prevent the baseline from being contaminated by weak-event years.

---

## 15. The base-state filtering step

This is one of the defining steps of SDC-map.

The slides state:

- the base state is filtered out from the spatiotemporal matrix,
- correlations are then computed from the scalar series and the filtered matrix,
- correlations are averaged over the selected positive or negative peaks.

What does this mean conceptually?

It means that SDC-map is not merely correlating the raw field with the scalar index. It first removes the background condition associated with normal years. So the final signal is closer to:

- anomaly relative to a baseline,
- event-conditioned response,
- local coupling after subtracting non-event structure.

This makes the maps more interpretable because they aim to isolate the spatial fingerprint of the selected episodes.

---

## 16. Correlation computation in SDC-map

Once peaks are chosen and the base state is filtered out, the method computes correlation maps.

The deck says these maps are computed for each:

- peak,
- grid point `(x0, y0)`,
- time lag, for example from `-12` to `+12` months.

So the loop structure is conceptually:

1. choose event class: positive or negative,
2. choose one selected peak,
3. choose one lag,
4. choose one grid point,
5. extract local field values in the correlation window `r_w`,
6. correlate those with the scalar series segment associated with the selected event,
7. store the resulting coefficient,
8. repeat over all grid points and lags,
9. then average across peaks.

The slides explicitly mention outputs related to:
1. correlations,
2. location of the peak,
3. lag,
4. “exact” location relative to the scalar series.

That suggests the method keeps track not only of map values, but also of the event timing metadata.

---

## 17. Significance testing in SDC-map

The presentation includes a dedicated significance slide.

### 17.1 Monte Carlo null distribution
A large ensemble of permutations, around 10,000 in the slide example, is used to generate alternative time series and simulate the correlation probability density function.

### 17.2 Tail-based significance
Significant correlations are assumed in the positive or negative tails of the simulated distribution. The slide also mentions use of a t-test.

The overall logic is:
- estimate what correlation values would be expected under a null arrangement,
- compare observed local correlations against that null,
- retain only those unlikely under the null.

This is especially important because local and lagged correlation analysis involves many comparisons and would otherwise be prone to false positives.

---

## 18. Outputs of SDC-map

The output is not just one map. It is a structured set of products.

### 18.1 Correlation maps by lag
For each lag, one can obtain a spatial map showing where the field is positively or negatively associated with the selected events.

### 18.2 Separate maps for positive and negative event classes
Positive peaks and negative peaks are analyzed separately.

### 18.3 Peak-averaged maps
After computing per-peak maps, the results are averaged across the `N+` or `N-` selected peaks.

### 18.4 Event metadata
The slides imply storage of:
- correlation value,
- peak identity,
- lag,
- exact temporal alignment relative to the scalar series.

### 18.5 A dynamic picture of teleconnection structure
By scanning lag, the user can infer the spatial evolution of the response through time.

This is exactly how the deck uses the method to show evolving responses to El Niño and other large-scale forcings.

---

## 19. What scientific questions SDC-map is designed for

From the applications shown, SDC-map is suited for questions like:

- Where does a climate mode imprint itself in atmospheric or surface fields?
- How does the spatial response evolve before, during, and after peak events?
- Are positive and negative phases spatially symmetric?
- Are droughts becoming larger, more energetic, or structured differently?
- Which regions act as hotspots or response corridors to large-scale forcing?
- How do regional impacts emerge from global drivers?

So SDC-map is not just a visualization method. It is a **diagnostic method for event-conditioned spatial teleconnections**.

---

## 20. Relation between SDC and SDC-map

The two methods are related, but not identical.

### SDC
- data type: two time series,
- purpose: find local, transient, scale-dependent coupling,
- output: local correlation pattern over time, lag, and scale.

### SDC-map
- data type: one scalar series plus one gridded field,
- purpose: find where and when the field responds to selected events in the scalar series,
- output: lagged spatial maps, often conditioned on positive and negative event classes.

A useful way to think about them is:

- **SDC** localizes coupling in time,
- **SDC-map** localizes coupling in time and space.

---

## 21. Why the method is powerful

The slide deck strongly implies several methodological strengths.

### 21.1 It handles nonstationarity
Relationships can turn on and off. SDC is built for that.

### 21.2 It handles delayed effects
Lag scanning is built in.

### 21.3 It handles scale dependence
Different window sizes reveal different structures.

### 21.4 It separates event phases
Positive and negative peaks can have different signatures.

### 21.5 It avoids overreliance on global averages
A weak global correlation does not imply absence of dynamical coupling.

### 21.6 It links pattern recognition to physically plausible mechanisms
The deck explicitly frames SDC as a tool to help assess whether couplings are physically plausible.

---

## 22. Main limitations and caveats

The slides are mostly methodological and promotional, so they emphasize strengths. But the method also has real caveats.

### 22.1 Parameter dependence
Results depend on:
- window length `s`,
- lag range,
- number of selected peaks `N+`, `N-`,
- base-state parameter `β`,
- correlation width `r_w`.

Poor parameter choices may overstate or hide structure.

### 22.2 Multiple comparisons
Scanning many times, scales, lags, and grid cells increases false-positive risk. The Monte Carlo step helps, but correction and interpretation still require care.

### 22.3 Correlation is not causation
The method detects structured association, not causal proof.

### 22.4 Autocorrelation and shared seasonality
Many environmental and disease signals are strongly autocorrelated and seasonal. Apparent local coupling can still emerge from shared structure unless carefully controlled.

### 22.5 Peak conditioning can be powerful but subjective
Selecting only the strongest events may isolate the mechanism, but it can also make results sensitive to event definition.

### 22.6 Interpretation requires domain knowledge
The method can reveal patterns, but physically plausible interpretation still depends on the science of the system.

---

## 23. How to explain SDC in one technical sentence

A concise technical description would be:

> SDC is a local, lagged, multiscale correlation framework that computes correlation on moving windows to detect transient and scale-specific coupling between time series.

And for SDC-map:

> SDC-map is the event-conditioned spatiotemporal extension of SDC that correlates a scalar index with a gridded field across space and lag, after selecting positive and negative peaks and filtering out a normal-state baseline.

---

## 24. Minimal pseudo-workflow for implementation

## 24.1 Temporal SDC pseudo-workflow

1. Choose two aligned time series `x(t)` and `y(t)`.
2. Select a window length `s`.
3. Select lag range `L`.
4. For each lag `ℓ in L`:
   1. Slide a window of length `s` along the record.
   2. Compute local correlation between the lagged windows.
5. Store `r(time, lag, scale)`.
6. Repeat for multiple `s`.
7. Visualize and interpret transient patches, sign changes, and lag structure.

## 24.2 SDC-map pseudo-workflow

1. Choose a scalar series `x(t)` and a gridded field `F(x, y, t)`.
2. Identify strongest positive peaks and strongest negative peaks in `x(t)`.
3. Define moderate years and normal years using a thresholding rule involving `β`.
4. Estimate the base-state field from normal years.
5. Remove the base state from `F`.
6. For each selected peak:
   1. For each lag `ℓ`:
      1. For each grid point `(x, y)`:
         1. Extract a local field segment of width `r_w`.
         2. Correlate with the scalar series segment aligned to the selected peak.
7. Build null distributions via Monte Carlo permutations.
8. Threshold by significance.
9. Average across peaks within positive and negative classes.
10. Produce lagged correlation maps.

---

## 25. Practical interpretation guide

When reading SDC results, I would interpret them as follows.

### If a relationship appears only at one scale
Treat cautiously. It may be real, but robustness is limited.

### If a relationship appears across neighboring scales
More convincing. That suggests a genuine scale band rather than an artifact of one window size.

### If the same sign appears repeatedly during known events
This is strong evidence of event-conditioned organization.

### If positive and negative peaks yield different maps
That implies asymmetry in the system response.

### If correlations are strongest at short lags only in high-activity periods
That suggests state-dependent sensitivity or threshold behavior.

### If global correlation is weak but local patterns are strong
That is exactly the kind of scenario SDC was designed to detect.

---

## 26. A very short intuitive analogy

A single full-series correlation is like averaging the sound of an entire song and asking whether two instruments are coordinated.

SDC instead listens in short overlapping fragments:
- sometimes the instruments are synchronized,
- sometimes one leads the other,
- sometimes one drops out,
- and this depends on the part of the song.

SDC-map adds the stage layout:
- not only when the coordination happens,
- but where in the orchestra it is expressed.

---

## 27. Bottom line

The slide deck presents SDC as a **transient-analysis tool** for systems in which couplings are intermittent, delayed, threshold-dependent, and scale-specific. The algorithm works by computing **local correlations on moving windows**, often across **multiple scales and lags**, to reveal hidden dynamical structure that global correlations miss.

The **SDC-map** extends the same philosophy to spatial fields. It selects positive and negative events in a scalar driver, defines and filters out a normal baseline, computes **lagged local correlation maps** at each grid point, and uses Monte Carlo methods to identify significant spatial fingerprints of the selected events.

In short:

- **SDC** answers: *when, at what scale, and with what lag are two time series coupled?*
- **SDC-map** answers: *where, when, and with what lag does a spatial field respond to selected events in a scalar series?*

---

## 28. References explicitly mentioned in the deck

The slides cite or mention:

- Rodó (1997; 2001)
- Rodríguez-Arias & Rodó (2004)
- Rodó & Rodríguez-Arias (2005)
- Rodó et al., PNAS (2002)
- Rodó et al. (2024)
- Fontal et al., Nature Computational Science (2021)

These appear to represent the methodological development of SDC and examples of its applications across climate, drought, cholera, influenza, and COVID-related analyses.

---

## 29. Suggested wording for reuse in documentation or a methods section

You can reuse or adapt this paragraph:

> Scale-Dependent Correlation (SDC) analysis is a local correlation framework designed to detect transient, lagged, and scale-specific coupling between nonstationary time series. Instead of computing a single correlation over an entire record, SDC evaluates correlation on moving windows of fixed length and, optionally, across multiple lags and temporal scales. This allows the identification of intermittent or threshold-dependent associations that would be diluted in full-series statistics. Its spatiotemporal extension, SDC-map, uses a scalar reference series together with a gridded field, selects strong positive and negative events in the scalar series, filters out a normal-state baseline, and computes lagged local correlation maps for each event and grid point. Significant responses are then identified through Monte Carlo testing and summarized as event-conditioned spatial fingerprints.

