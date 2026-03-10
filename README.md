This is an empirical implementation of the Mutual Information Neural Estimator (MINE), as proposed by Belghazi et al. (2018). While traditional mutual information ($I(X;Y)$) estimation relies on methods that struggle in higher dimensions, MINE leverages the Donsker-Varadhan representation of the KL-divergence to turn estimation into a dual-optimization problem. The Dosker-Varadhan representation is given by :

\[D_{KL}(\mathbb{P}\,\|\,\mathbb{Q}) = \sup_{T:\Omega \to \mathbb{R}}\left\{\mathbb{E}_{\mathbb{P}}[T] -\log\big(\mathbb{E}_{\mathbb{Q}}[e^{T}]\big)\right\}\]

The objective:
I applied MINE to a Single-Input Single-Output (SISO) Communication Channel to observe how neural networks learn fundamental information theoretic limits. Specifically, I estimated the capacity of an AWGN Channel under three different signaling regimes:
*** Gaussian Signaling: The theoratical Shannon Limit is given by $C = \frac{1}{2}\log_2(1 + \text{SNR})$.
*** BPSK Modulation: A constellation with a 1 bit entropy ceiling .
*** QPSK Modulation: A constellation with a 2-bit entropy ceiling .

