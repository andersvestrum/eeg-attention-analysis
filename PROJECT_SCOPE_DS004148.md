# Project Scope: ds004148

## Working Title
Using EEG bandpower features to study proxy states of reduced engagement and focused cognition

## Motivation
Our original class idea was: "Using EEG signal to measure when the brain gets bored of repeated images."

There is not a public EEG dataset that directly labels "boredom while viewing repeated images." To keep the project scientifically defensible, we are reframing boredom as a proxy problem:

- lower-engagement or idle EEG states
- higher-focus or cognitively effortful EEG states

This still supports the broader question of attention drop under repetitive stimulation, but avoids overclaiming that we directly measured boredom.

## Main Dataset
We use OpenNeuro dataset `ds004148`:

- resting states: `eyesclosed`, `eyesopen`
- cognitive states: `mathematic`, `memory`, `music`

All conditions are recorded within the same dataset, which is methodologically cleaner than training across unrelated datasets.

## First Analysis
We narrow the first binary comparison to:

- `eyesclosed` -> low-engagement proxy
- `mathematic` -> high-engagement proxy

To keep the download and runtime manageable, we start with `session1`.

## Feature Logic
For each EEG recording, we split the signal into fixed windows and compute:

- `theta_power` (4-7 Hz)
- `alpha_power` (8-12 Hz)
- `beta_power` (13-30 Hz)
- relative power for each band
- `alpha_beta_ratio`
- `beta_alpha_ratio`
- `theta_beta_ratio`

These are standard spectral summaries, not medical diagnoses.

## Claim We Can Defend
We are not claiming to detect boredom directly.

We are claiming that EEG spectral features can distinguish lower-engagement resting states from higher-demand cognitive states, and that this is a first step toward studying boredom-like attention drop during repetitive visual exposure.

## Immediate Deliverable
Produce:

- a processed feature table from `ds004148`
- boxplots of key features
- an alpha-vs-beta scatter plot
- a simple logistic regression confusion matrix

## Future Extension
If time allows, we can expand to:

- `eyesopen` vs `memory`
- multi-class condition comparison across all five states
- time-varying engagement score within recordings
