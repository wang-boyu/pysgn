---
title: 'PySGN: A Python package for constructing synthetic geospatial networks'
tags:
  - synthetic geospatial networks
  - python
  - spatial simulation
  - complex systems
authors:
  - name: Boyu Wang
    orcid: 0000-0001-9879-2138
    corresponding: true
    affiliation: 1
  - name: Andrew Crooks
    orcid: 0000-0002-5034-6654
    corresponding: false
    affiliation: 1
  - name: Taylor Anderson
    orcid: 0000-0003-1145-0608
    corresponding: false
    affiliation: 2
  - name: Andreas Züfle
    orcid: 0000-0001-7001-4123
    corresponding: false
    affiliation: 3
affiliations:
 - name: Department of Geography, University at Buffalo, Buffalo, New York, USA
   index: 1
 - name: Geography & Geoinformation Science Department, George Mason University, Fairfax, Virginia, USA
   index: 2
 - name: Department of Computer Science, Emory University, Atlanta, Georgia, USA
   index: 3
date: 1 January 2024
bibliography: paper.bib
doi: 10.3847/xxxxx

---

# Summary

PySGN (**Py**thon for **S**ynthetic **G**eospatial **N**etworks) is an open-source Python package designed to construct synthetic undirected geospatial networks that consider the coordinates of nodes. By incorporating the spatial information, the package extends classic network generation models, such as Erdős-Rényi [@erdos1960evolution], Watts-Strogatz [@watts1998collective] and Barabási–Albert [@barabasi1999emergence]. It creates synthetic geospatial networks that mimic the spatial relationships observed in real-world networks (e.g., decaying connectivity with respect to node distance) and can be used for various simulation and analysis tasks. PySGN integrates with the PyData ecosystem, utilizing libraries, such as GeoPandas [@joris_van_den_bossche_2024_12625316] and NetworkX [@osti_960616], to facilitate the creation, utilization, and analysis of synthetic geospatial networks.

PySGN offers flexibility in parametrization, allowing users to define networks by an average node degree or expected degree for each node. Users can also define custom constraints and control the rewiring probability, distance decay, and other key parameters, making it adaptable to various research contexts. PySGN is intended for researchers and practitioners in fields, such as urban planning, epidemiology, and social science, who require robust tools for simulating and analyzing complex geospatial networks.

# Statement of Need

The need for synthetic geospatial networks arises from their utility in social simulations, including modeling of transportation systems, pedestrian movements, and the spread of infectious diseases [@zufle2024silico]. Traditional synthetic populations often lack the integration of geographic social networks, which are crucial for accurately capturing social connections and spatial dynamics, to explore the effects of spatial proximity on social interactions, mobility patterns, and network robustness [@jiang2024large].

PySGN addresses this gap by providing a tool that not only generates geographically explicit networks but also incorporates key network properties, such as clustering, preferential attachment and spatial decay. These features allow users to explore different network properties and configurations (e.g., average node degree). This is essential for a variety of simulation scenarios, where understanding spatial relationships and social dynamics is critical for analyzing and modeling complex systems. This makes PySGN suitable for diverse applications, including infrastructure resilience studies, agent-based modeling, and geospatial data analysis.

## Related Work and Existing Tools

Several open‑source Python packages handle spatial networks or random network generation. PySGN focuses on synthesizing spatially explicit networks, which differs from these packages in scope and functionality.

The `neatnet` package provides tools for pre‑processing street network geometry aimed at simplifying complex road geometries (e.g., removing dual carriageways and roundabouts). It simplifies existing street networks to create morphological representations of street space rather than generating new synthetic networks [@fleischmann2026adaptive]. 

`OSMnx` is a Python package that allows users to download, model, analyze and visualize street networks and other geospatial features from OpenStreetMap. With a single line of code, users can obtain walking, driving or biking networks and analyze or visualize them [@boeing2025osmnx]. However, OSMnx does not provide facilities for generating synthetic networks or modifying the spatial distribution of nodes. 

Within the PySAL ecosystem, the `spaghetti` library is an open‑source package for the analysis of network‑based spatial data that originated from the network module in PySAL and is designed to build and analyze graph‑theoretic networks and network events [@gaboardi2021spaghetti]. The PySAL team is currently transitioning functionality from `spaghetti` to the experimental `libpysal.graph` module, whose `Graph` class encodes spatial weights matrices and is still considered experimental [@pysal2007]. These packages provide tools for analyzing existing spatial networks and building weights matrices but do not  generate synthetic geospatial networks; their focus differs from the generative models implemented in PySGN. 

General‑purpose network libraries such as NetworkX and igraph include random graph generators. NetworkX offers functions to generate Erdős–Rényi, Watts–Strogatz and Barabási–Albert graphs [@osti_960616], and igraph includes similar random graph models (Barabási–Albert, Erdős–Rényi, Watts–Strogatz and other stochastic graph models) [@csardi2006igraph]. However, these models do not incorporate geographic information or spatial distances: nodes are considered abstract or are uniformly distributed, and edge probabilities do not depend on spatial proximity. Consequently, while NetworkX and igraph are excellent tools for general network analysis and for generating abstract random graphs, they are unsuitable for constructing geospatial networks where spatial proximity influences connectivity. 

By contrast, PySGN synthesizes geospatial networks by embedding nodes in geographic coordinate space and incorporating distance‑decay functions and other constraints into the generation process. It extends the classical random graph models (Erdős–Rényi, Watts–Strogatz and Barabási–Albert) to spatial contexts and integrates with GeoPandas and NetworkX to provide geospatially explicit network generation and analysis. Thus, PySGN fills a gap between packages that simplify or analyze existing spatial networks and those that generate abstract random graphs. 

# Acknowledgements

The algorithms implemented in PySGN are based on the work of @alizadeh2017generating and @10.1145/3615896.3628345, with several improvements and modifications, including bug fixes, performance enhancements, and additional features. We would like to thank the authors for their contributions to the field of synthetic geospatial network generation.

This work was supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0423C0025. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

# References