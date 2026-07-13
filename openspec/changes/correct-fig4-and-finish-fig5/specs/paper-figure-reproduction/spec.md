## MODIFIED Requirements

### Requirement: Final-batch-size sweep figure (paper Fig. 4)
The system SHALL provide a script that plots PTSBE throughput (shots/s) vs. g for multiple final batch sizes `bf` (e.g. 24, 26, 28) **at n=200**, matching the paper's own Fig. 4 configuration (Sec. V-A: "This effect of sample efficiency by final batch-size bf is demonstrated for n = 200 systems in Fig. 4"), from benchmark results covering those `bf` values across multiple g values. Data collected at other n values MAY be plotted separately for exploratory purposes but SHALL NOT be presented as a reproduction of paper Fig. 4 without noting the n mismatch.

#### Scenario: Figure generated from bf sweep results
- **WHEN** the figure script is run against results covering `bf` in {24, 26, 28} at n=200, across multiple g values
- **THEN** a plot is produced with g on the x-axis, throughput on the y-axis (log scale), one line per `bf`

#### Scenario: Non-paper-n data is not presented as the paper's Fig. 4
- **WHEN** bf-sweep results exist at an n other than 200 (e.g. from exploratory runs)
- **THEN** any figure or summary text built from that data explicitly notes it is not the paper's tested configuration, rather than being labeled or discussed as if it reproduces Fig. 4
