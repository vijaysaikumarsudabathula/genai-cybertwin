# -*- coding: utf-8 -*-
import docx
from build_helpers import add_heading1, add_reference, OUT

doc = docx.Document(str(OUT))

add_heading1(doc, "REFERENCES")

refs = [
    "[1] S. Wachter, B. Mittelstadt, and C. Russell, \u201cCounterfactual explanations without opening "
    "the black box: Automated decisions and the GDPR,\u201d Harvard J. Law & Technology, vol. 31, no. 2, "
    "pp. 841\u2013887, 2018.",

    "[2] R. K. Mothilal, A. Sharma, and C. Tan, \u201cExplaining machine learning classifiers through "
    "diverse counterfactual explanations,\u201d in Proc. 2020 Conf. Fairness, Accountability, and "
    "Transparency (FAT* '20), Barcelona, Spain, 2020, pp. 607\u2013617. doi: 10.1145/3351095.3372850.",

    "[3] S. M. Lundberg and S.-I. Lee, \u201cA unified approach to interpreting model predictions,\u201d "
    "in Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017, pp. 4765\u20134774.",

    "[4] S. M. Lundberg et al., \u201cFrom local explanations to global understanding with explainable "
    "AI for trees,\u201d Nat. Mach. Intell., vol. 2, no. 1, pp. 56\u201367, Jan. 2020. "
    "doi: 10.1038/s42256-019-0138-9.",

    "[5] T. Chen and C. Guestrin, \u201cXGBoost: A scalable tree boosting system,\u201d in Proc. 22nd "
    "ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), 2016, pp. 785\u2013794. "
    "doi: 10.1145/2939672.2939785.",

    "[6] L. Breiman, \u201cRandom forests,\u201d Mach. Learn., vol. 45, no. 1, pp. 5\u201332, Oct. 2001. "
    "doi: 10.1023/A:1010933404324.",

    "[7] R. Storn and K. Price, \u201cDifferential evolution \u2013 a simple and efficient heuristic for "
    "global optimization over continuous spaces,\u201d J. Global Optim., vol. 11, no. 4, pp. 341\u2013359, "
    "1997. doi: 10.1023/A:1008202821328.",

    "[8] R. Sommer and V. Paxson, \u201cOutside the closed world: On using machine learning for network "
    "intrusion detection,\u201d in Proc. 2010 IEEE Symp. Security and Privacy, 2010, pp. 305\u2013316. "
    "doi: 10.1109/SP.2010.25.",

    "[9] A. L. Buczak and E. Guven, \u201cA survey of data mining and machine learning methods for "
    "cyber security intrusion detection,\u201d IEEE Commun. Surv. Tuts., vol. 18, no. 2, pp. "
    "1153\u20131176, 2016. doi: 10.1109/COMST.2015.2494502.",

    "[10] R. Alenezi and S. A. Ludwig, \u201cExplainability of cybersecurity threats data using SHAP,\u201d "
    "in Proc. 2021 IEEE Symp. Series Comput. Intell. (SSCI), Orlando, FL, USA, 2021, pp. 1\u201310. "
    "doi: 10.1109/SSCI50451.2021.9659888.",

    "[11] Z. Ji et al., \u201cSurvey of hallucination in natural language generation,\u201d ACM Comput. "
    "Surv., vol. 55, no. 12, art. 248, pp. 1\u201338, Dec. 2023. doi: 10.1145/3571730.",

    "[12] E. Tabassi, \u201cArtificial Intelligence Risk Management Framework (AI RMF 1.0),\u201d NIST AI "
    "100-1, National Institute of Standards and Technology, Gaithersburg, MD, USA, Jan. 2023. "
    "doi: 10.6028/NIST.AI.100-1.",

    "[13] M. T. Ribeiro, S. Singh, and C. Guestrin, \u201c\u2018Why should I trust you?\u2019: Explaining the "
    "predictions of any classifier,\u201d in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and "
    "Data Mining (KDD), 2016, pp. 1135\u20131144. doi: 10.1145/2939672.2939778.",

    "[14] N. Moustafa and J. Slay, \u201cUNSW-NB15: A comprehensive data set for network intrusion "
    "detection systems,\u201d in Proc. 2015 Military Commun. and Inf. Syst. Conf. (MilCIS), 2015, pp. "
    "1\u20136. doi: 10.1109/MilCIS.2015.7348942.",

    "[15] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, \u201cA detailed analysis of the KDD CUP "
    "99 data set,\u201d in Proc. 2009 IEEE Symp. Comput. Intell. Security and Defense Appl. (CISDA), "
    "2009, pp. 1\u20136. doi: 10.1109/CISDA.2009.5356528.",

    "[16] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, \u201cSMOTE: Synthetic minority "
    "over-sampling technique,\u201d J. Artif. Intell. Res., vol. 16, pp. 321\u2013357, 2002. "
    "doi: 10.1613/jair.953.",

    "[17] H. He and E. A. Garcia, \u201cLearning from imbalanced data,\u201d IEEE Trans. Knowl. Data Eng., "
    "vol. 21, no. 9, pp. 1263\u20131284, Sep. 2009. doi: 10.1109/TKDE.2008.239.",

    "[18] Forum of Incident Response and Security Teams (FIRST), \u201cCommon Vulnerability Scoring "
    "System v3.1: Specification document,\u201d 2019. [Online]. Available: "
    "https://www.first.org/cvss/v3-1/cvss-v31-specification_r1.pdf.",
]
for ref in refs:
    add_reference(doc, ref)

doc.save(str(OUT))
print("References done. Final paragraph count:", len(doc.paragraphs))
