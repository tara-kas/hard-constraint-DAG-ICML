# hard-constraint-DAG-ICML
Scalable Hard-Constraint Projection for DAG Ontologies in Neural Classification

Problem statement: Real-world knowledge is structured in direct acyclic graphs (DAGs) rather than trees. In the application of healthcare, a “viral pneumonia” is both a “viral infection” and a “respiratory disease”. A child word can have multiple given parents. Major ontological datasets such as Gene Ontology (45,000 concepts) or SNOMED-CT (350,000 concepts) are explicit DAGs (as opposed to trees forcefully formatted into DAGs) with heavy multiple-inheritance. Neural classifiers ignore this structure and are prone to producing logically fallible outputs, like a high confidence for “viral pneumonia” but a low confidence for “respiratory disease”. This violates semantics of the inherent ontology which can undermine credibility and trust in critical domains such as medical diagnosis.

Test reports:
- violation_rate_percent     → % of edges violating constraints
- violations_per_sample      → Average violations per prediction
- samples_with_violations    → % of samples with ≥1 violation
- max_violation_magnitude    → Worst-case difference