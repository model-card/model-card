"""
====================================
Model Card for a Classification Task
====================================
This example shows how you can create a model card for a classification task,
and attach concerns and fairness related metrics to the card.

This example requires `fairlearn`.
"""
# %%
# Here we have all the imports required to run the rest of the script
import uuid
import json
import numpy as np
import tempfile
import webbrowser
import sklearn.metrics as skm
from datetime import date
from numpy.random import RandomState
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from fairlearn.metrics import MetricFrame
from modelcard import ModelCard

# %%
# Now we are going to create some synthetic data for our hypothetical promotion
# distribution scenario. The hypothetical problem here is: should we give this
# customer a promotion (a discount) or not. We assume three different processes
# generating the data for three different gender categories: women, men, and
# unspecified. We intentionally generate the data from different background
# distributions to reflect different cultural parameters governing different
# cultural gender norms and behaviors. We also add more noise to some of them
# than others, to reflect how certain processes can cause the data for a
# certain category to be more noisy and less accurate than others (think biases
# etc).

rng = RandomState(seed=42)

X_women, y_women = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=4,
    n_classes=2,
    class_sep=1,
    random_state=rng,
)

X_men, y_men = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=4,
    n_classes=2,
    class_sep=2,
    random_state=rng,
)

X_unspecified, y_unspecified = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=4,
    n_classes=2,
    class_sep=0.5,
    random_state=rng,
)

X = np.r_[X_women, X_men, X_unspecified]
y = np.r_[y_women, y_men, y_unspecified]
gender = np.r_[["Woman"] * 500, ["Man"] * 500, ["Unspecified"] * 500].reshape(
    -1,
)

X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
    X, y, gender, test_size=0.3, random_state=rng
)

# %%
# Now we train a classifier on the data
clf = make_pipeline(PCA(n_components=4), HistGradientBoostingClassifier())
clf.fit(X_train, y_train)

# %%
# We can finally check the performance of our model for different groups.
# First we look at the soft values and see how average precision and roc-auc
# are among different groups.
y_pred = clf.predict_proba(X_test)[:, 1]
multi_metric_soft = MetricFrame(
    {
        "average precision": skm.average_precision_score,
        "roc-auc": skm.roc_auc_score,
        "count": lambda x, y: len(x),
    },
    y_test,
    y_pred,
    sensitive_features={"gender": gender_test},
)

print("Scores per group:\n", multi_metric_soft.by_group)
print("Overall scores:\n", multi_metric_soft.overall)

# %%
# Having the discrete predictions/classes, we can also calculate metrics such
# as precision, recall, and f1 scores, and look at slices of data based on our
# groups.
y_pred = clf.predict(X_test)
multi_metric_hard = MetricFrame(
    {
        "precision": skm.precision_score,
        "recall": skm.recall_score,
        "f1": skm.f1_score,
    },
    y_test,
    y_pred,
    sensitive_features={"gender": gender_test},
)

print("Scores per group:\n", multi_metric_hard.by_group)
print("Overall scores:\n", multi_metric_hard.overall)

# %%
# Now that we have all the data, we can also create a model card for our new
# shiny model and use-case. Think of a model card as a one pager explaining
# what the model does, and what its limitations are.

model_card = ModelCard()

# %%
# `model_details` section is where we explain the overall purpose of the model.
# We also attach owners and references here so that people who read the card
# can contact the relevant teams/people.

# The first one is the model name. This name should be specific enough so that
# you can understand what it does, but it should ideally not include the date
# or version of the model if the only thing changed is that the model is
# retrained on new data. That information should go in the `version`. If adding
# or removing features or changing the architecture of the model is recorded in
# a ticket or an issue, you can leave the number of the issue here.
model_card.model_details.name = "Promotion Offering - 212"

# The overview should give the reader an idea of what the model does, and what
# kinds of data are used to train the model.
model_card.model_details.overview = (
    "This model predicts whether giving a user a promotion code would "
    "significantly increase their chances of putting a purchase. We use "
    "customer's purchase history as well as information we have about the "
    "market and click history to make the prediction."
)

# Here you can leave the contact details of the team and/or people who can be
# contacted in case of questions.
model_card.model_details.owners = [
    {
        "name": "Campaign Prediction Team",
        "contact": "campaign-predictoin@example.com",
    },
    {"name": "Jane Doe", "contact": "jane.dow@example.de"},
]

# References can be futher documentation, scientific publications, your own
# documentation pages, etc. Anything which would give the reader more context
# about the model.
model_card.model_details.references = [
    "https://docs.example.com/campaigns.html",
    "https://dl.acm.org/doi/abs/10.1145/2939672.2939785",
]

# If there is a way you want people to cite your model or work, you can put it
# in this field.
model_card.model_details.citation = (
    "Jane Doe, Improving User Convergence Using Their Behavior, "
    "Campaign Prediction Team, Example Company, "
    "https://docs.example.com/campaigns.html"
)

# You should also attach a unique version ID to this model card. You may be
# retraining the model periodically, but not changing the logic behind the
# model, which means most of the model card would stay the same, and only the
# later sections would change. You may be using a backend which would store all
# your cards and then show the changes over time for different versions.
model_card.model_details.version.name = str(uuid.uuid4())
model_card.model_details.version.date = str(date.today())


# %%
# `considerations` section of the model card makes sure users and readers
# understand the limitations and ethical considerations of the model, and when
# to avoid using the model or the output of the model in their use cases.

# the ethical considerations are a broad umbrella category for anything which
# should be considered and is thought of when we think of responsible and
# ethical AI.
model_card.considerations.ethical_considerations = [
    {
        "name": (
            "Location Data: we use IP address ranges as a proxy for the "
            "user's location. This can fail if the user is using a VPN for "
            "instance which would mask their real IP address."
        ),
        "mitigation_strategy": (
            "Detect VPN addresses and if the user is connected using a VPN, "
            "use the domain they're browsing (eg. DE) and use a special value "
            "representing the whole DE region instead."
        ),
    },
    {
        "name": (
            "Uncalibrated Soft Outputs: our model gives a soft score as well "
            "as the final {YES, NO} predictions. These values are used for "
            "debugging purposes and should not be treated as scores."
        ),
        "mitigation_strategy": (
            "These outputs should be callibrated before used as scores. We "
            "have this callibration in our roadmap, for now the model users "
            "should do that themselves if they wish to use these soft values."
        ),
    },
    {
        "name": (
            "Effect on Minority Groups: we have not been able to check "
            "whether our model has any unwanted biases regarding all relevant "
            "aspects of our customers. We have analyzed our model for any "
            "biases regarding gender, and eventhough we have not used gender "
            "in our model as input, there are clear disparities. Regarding "
            "other aspects such as race, we don't have the data to check for "
            "them, and more work is needed for us to assess the risks."
        ),
        "mitigation_strategy": (
            "We are working on improving the situation on this aspect. For "
            "now it's important to remember these biases when using the "
            "output of the model."
        ),
    },
]

# We should be very clear on where the model should not be used and what the
# limitations are. Sometimes downstream developers may use the output of our
# models in ways which are outside the scope and analysis of the model.
model_card.considerations.limitations = [
    "The model is only accurate if there are enough data about the user. "
    "Therefore it should only be used for customers who have had at least two "
    "purchases in the past and their current session is at least 5 clicks "
    "long.",
    "The output of this model has nothing to do with the value of the "
    "customer, and should not be treated and used as such.",
]

# Sometimes a model may end up being used by many teams at your company, and
# it's important to be clear as what the use-cases for the model are.
model_card.considerations.use_cases = [
    "Predicting whether or not offering the user a coupon would convince them "
    "to put a purchase."
]

# This list may not be exhaustive, but it's good practice to keep this list as
# accurate as possible. Understandably if you're registering your model's
# output somewhere e.g. writing them on an the cloud, other teams may get
# access to those data without your knowledge. So keep this list up to date
# as much as you can, with the understanding that it may not be accurate.
model_card.considerations.users = [
    "The team which directly uses the outputs",
    "The other team which uses the outputs for some analysis about our "
    "campaigns.",
]

# Especially when there are certain risks, which is usually the case, we should
# be conscious about those risks and the tradeoffs we're making when using our
# solutions.
model_card.considerations.tradeoffs = [
    "As pointed out in the ethical considerations, there may be certain "
    "biases that ideally should not be there. These biases mean we would "
    "treat our customers differently based on some of their characteristics. "
    "This poses both an ethical and a legal risk for us, and we are making a "
    "conscious decision to take these risk while working on solutions to fix "
    "them. Using this model is resulting in a level of customer satisfaction "
    "that we're willing to take the risks for it."
]

# %%
# `model_architecture` section includes information about the model itself,
# as well sa the train and test data used in preparing the model.

# Here you can explain the architecture and the algorithms used in your model.
model_card.model_parameters.model_architecture = (
    "This model uses a principal component analysis (PCA) to reduce the "
    "number of dimensions, then feed that to a HistGradientBoostingClassifier "
    "from scikit-learn which is an implementation of the algorithm used in "
    "xgboost."
)


# Next, you can include information about the train and the validation set,
# and you can also include images here. Please refer to [model card tookit's
# guide](https://www.tensorflow.org/responsible_ai/model_card_toolkit/examples/Scikit_Learn_Model_Card_Toolkit_Demo)
# on how to add graphics here.

# This section also gives general information about the data, and its format.
model_card.model_parameters.data.train.name = "Campaign prediction dataset"

# Ideally we should be able to point readers to a place where they can find
# more information about the data, and the data itself. If your data governance
# has provided a place where there are information about datasets, you can use
# those here.
model_card.model_parameters.data.train.link = (
    "https://data-explanations.com/asset/my-data-id"
)

# The data includes information using which users can be personally identified
# and therefore the sensitive flag is True.
model_card.model_parameters.data.train.sensitive = True

# And finally, we can talk about the input and output format of the model. In
# reality, this can be more specific, especially if the input is things other
# than just numerical values.
model_card.model_parameters.input_format = (
    "The model itself takes the inputs in the form of a numpy array, but our "
    "API accepts the same values through a JSON payload."
)

# This should give enough information to readers on how they can understand
# the output from the model.
model_card.model_parameters.output_format = (
    "Depending on the requested output type, we give either the predicted "
    "class as {YES/NO}, or also return the soft value taken from the "
    "`predict_proba` method, which gives an uncalibrated probability for each "
    "class."
)

# %%
# `quantitative_analysis` section includes concrete metrics about the data and
# the performance of the model, as well as fairness related metrics or metrics
# sliced according to different groups.

# Like the data section, you can attach graphics here.

# We need to convert the output we get from MetricFrame to something we can put
# in the model card. This helper function does that for us.


def convert_metric_frame(metric_frame):
    """Convert a fairlearn.metrics.MetricFrame to model card metrics."""
    metrics = []
    for metric, values in metric_frame.by_group.to_dict().items():
        for slice, value in values.items():
            metrics.append({"type": metric, "slice": slice, "value": value})

    for metric, value in metric_frame.overall.to_dict().items():
        metrics.append({"type": metric, "value": value})

    return metrics


metrics = convert_metric_frame(multi_metric_soft)
metrics.extend(convert_metric_frame(multi_metric_hard))

model_card.quantitative_analysis.performance_metrics = metrics

# %%
# At this point you can immediately have a JSON version of your model card.
payload = model_card.to_json()

print(payload)

# Then we take the generated html document, and launch a browser tab to view
# the document.
# html = mct.export_format()

# with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
#     url = "file://" + f.name
#     f.write(html)
# webbrowser.open(url)
