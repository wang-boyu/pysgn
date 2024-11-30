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

PySGN is an open-source Python package designed to construct synthetic undirected geospatial networks that consider the coordinates of nodes. The package extends classic network generation models, such as Erdős-Rényi [@erdos1960evolution] and Watts-Strogatz [@watts1998collective]. It includes spatial components, creating synthetic geospatial networks that mimic the spatial relationships observed in real-world networks. PySGN integrates with the PyData ecosystem, utilizing libraries, such as GeoPandas [@joris_van_den_bossche_2024_12625316] and NetworkX [@osti_960616], to facilitate the creation, utilization, and analysis of synthetic geospatial networks.

PySGN offers flexibility in parametrization, allowing users to define networks by an average node degree or expected degree for each node. Users can also define custom constraints and control the rewiring probability, distance decay, and other key parameters, making it adaptable to various research contexts. PySGN is intended for researchers and practitioners in fields, such as urban planning, epidemiology, and social science, who require robust tools for simulating and analyzing complex geospatial networks.

# Statement of Need

The need for synthetic geospatial networks arises from their utility in social simulations, including modeling of transportation systems, pedestrian movements, and the spread of infectious diseases. Traditional synthetic populations often lack the integration of geographic social networks, which are crucial for accurately capturing social connections and spatial dynamics, to explore the effects of spatial proximity on social interactions, mobility patterns, and network robustness.

PySGN addresses this gap by providing a tool that not only generates geographically explicit networks but also incorporates key network properties, such as clustering, preferential attachment and spatial decay. These features allow users to explore different network properties and configurations (e.g., average node degree). This is essential for a variety of simulation scenarios, where understanding spatial relationships and social dynamics is critical for analyzing and modeling complex systems. This makes PySGN suitable for diverse applications, including infrastructure resilience studies, agent-based modeling, and geospatial data analysis.

# Acknowledgements

The algorithms implemented in PySGN are based on the work of @alizadeh2017generating and @10.1145/3615896.3628345, with several improvements and modifications.

# References