# Dataset review

Review date: 20 November 2026.

## Candidate review

- **AMAG maintenance records described in Bolla et al. (2018):** The study reports car odometer data at AMAG garage maintenance visits and about 877,000 complete mileage-period samples derived from roughly 300,000 cars. Those records were provided by AMAG; the publication is a research paper, not a dataset release. No downloadable file or reuse license was located. Personal/location-related attributes are described, so redistributing records would require provenance and privacy review. Not used.
- **Kaggle Vehicle Maintenance Data:** A later paper describes a 50,000-row, 20-column dataset with binary `Need_Maintenance` target and mixed vehicle types, and third-party project docs cite the Kaggle page. This target is classification, not remaining service kilometres. The Kaggle dataset page/download and license were not independently accessible for verification, so no file or license is claimed here. Not used.
- **UCI Car Evaluation:** The official repository reports 1,728 rows, six categorical features and a car acceptability target derived from a decision model. Although `maint` is one input attribute, there are no service dates, odometers, service intervals or remaining-km labels. It is licensed CC BY 4.0 but unsuitable for this regression problem. Not used.
- **TRODO odometer dataset:** Contains annotated odometer images for computer vision, not service records. Not used.
- **VIUS and transit maintenance data:** These describe broader vehicle inventory/use or transit revenue-vehicle maintenance, not passenger-car service interval outcomes. Not used.

## Adopted alternative

Because the reviewed candidates do not supply verified car-only continuous remaining-service labels with clear reuse rights, the project uses a deterministic synthetic generator. The checked-in CSV has 8,000 rows, 11 columns, no missing cells and no exact duplicate rows. Target values and relationship assumptions are defined in `src/dataset.py` and `dataset/README.md`. Test metrics evaluate agreement with those synthetic assumptions only; they do not estimate field accuracy.

## Sources

1. Bolla, M., Held, M., Küng, L., Georges, G., & Boulouchos, K. (2018). *Vehicle motion patterns for energy research: Comparison of annual mileage using vehicle- and person-based data*. Swiss Transport Research Conference. https://www.strc.ch/2018/Bolla_EtAl.pdf
2. UCI Machine Learning Repository. *Car Evaluation*. https://archive.ics.uci.edu/dataset/19/car+evaluation
3. Mouheb, K., et al. (2025). *A Comprehensive Study on Predicting the Need for Vehicle Maintenance Using Machine Learning*. Engineering Proceedings, 107(1), 89. https://www.mdpi.com/2673-4591/107/1/89
4. Kaggle. *Vehicle Maintenance Data*. Page referenced by the 2025 study: https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data. The original download and license were not independently verified; this is recorded as an excluded candidate, not a dataset source for the model.
