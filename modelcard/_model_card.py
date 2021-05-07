# This file has been modified by the model card team from the original, which
# is licensed as follows:
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model Card Data Class.

The Model Card (MC) is the document designed for transparent reporting of AI
model provenance, usage, and ethics-informed evaluation. The model card can be
presented by different formats (e.g. HTML, PDF, Markdown). The properties of
the Model Card (MC) are defined by a json schema. The ModelCard class in the
ModelCardsToolkit serves as an API to read and write MC properties by the
users.
"""

import abc
import json as json_lib
from typing import Any, Dict, List, Optional, Text

import dataclasses
from .utils import validate_json_schema
from .utils import get_latest_schema_version

_SCHEMA_VERSION_STRING = "schema_version"


# TODO(b/181702622): Think about a smart and clean way to control the required
# field.
class BaseModelCardField(abc.ABC):
    """Model card field base class.

    This is an abstract class. All the model card fields should inherit this
    class.
    """

    def to_json(self) -> Text:
        """Convert this class object to json."""
        return json_lib.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[Text, Any]:
        """Convert your model card to a python dictionary."""
        # ignore None properties recursively to allow missing values.
        def ignore_none(properties):
            {k: v for k, v in properties if v}

        return dataclasses.asdict(self, dict_factory=ignore_none)

    def clear(self):
        """Clear the subfields of this BaseModelCardField."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, BaseModelCardField):
                field_value.clear()
            elif isinstance(field_value, list):
                setattr(self, field_name, [])
            else:
                setattr(self, field_name, None)

    @classmethod
    def _get_type(cls, obj: Any):
        return type(obj)


@dataclasses.dataclass
class Owner(BaseModelCardField):
    """The information about owners of a model.

    Attributes:
      name: The name of the model owner.
      contact: The contact information for the model owner or owners. These could
        be individual email addresses, a team mailing list expressly, or a
        monitored feedback form.
    """

    name: Optional[Text] = None
    contact: Optional[Text] = None


@dataclasses.dataclass
class Version(BaseModelCardField):
    """The information about versions of a model.

    If there are multiple versions of the model, or there may be in the future,
    it’s useful for your audience to know which version of the model is
    discussed
    in the Model Card. If there are previous versions of this model, briefly
    describe how this version is different. If no more than one version of the
    model will be released, this field may be omitted.

    Attributes:
      name: The name of the version.
      date: The date this version was released.
      diff: The changes from the previous version.
    """

    name: Optional[Text] = None
    date: Optional[Text] = None
    diff: Optional[Text] = None


@dataclasses.dataclass
class License(BaseModelCardField):
    """The license information for a model.

    Attributes:
      identifier: A standard SPDX license identifier
        (https://spdx.org/licenses/), or "proprietary" for an unlicensed
        Module.
      custom_text: The text of a custom license.
    """

    identifier: Optional[Text] = None
    custom_text: Optional[Text] = None


@dataclasses.dataclass
class Reference(BaseModelCardField):
    """Reference for a model.

    Attributes:
      reference: A reference to a resource.
    """

    reference: Optional[Text] = None


@dataclasses.dataclass
class Citation(BaseModelCardField):
    """A citation for a model.

    Attributes:
      style: The citation style, such as MLA, APA, Chicago, or IEEE.
      citation: the citation.
    """

    style: Optional[Text] = None
    citation: Optional[Text] = None


@dataclasses.dataclass
class ModelDetails(BaseModelCardField):
    """This section provides a general, high-level description of the model.

    Attributes:
      name: The name of the model.
      overview: A description of the model card.
      documentation: A more thorough description of the model and its usage.
      owners: The individuals or teams who own the model.
      version: The version of the model.
      licenses: The license information for the model. If the model is licensed
        for use by others, include the license type. If the model is not
        licensed for future use, you may state that here as well.
      references: Provide any additional links the reader may need. You can
        link to foundational research, technical documentation, or other
        materials that may be useful to your audience.
      citations: How should the model be cited? If the model is based on
        published academic research, cite the research.
    """

    name: Optional[Text] = None
    overview: Optional[Text] = None
    documentation: Optional[Text] = None
    owners: List[Owner] = dataclasses.field(default_factory=list)
    version: Optional[Version] = dataclasses.field(default_factory=Version)
    licenses: List[License] = dataclasses.field(default_factory=list)
    references: List[Reference] = dataclasses.field(default_factory=list)
    citations: List[Citation] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Graphic(BaseModelCardField):
    """A named inline plot.

    Attributes:
      name: The name of the graphic.
      image: The image string encoded as a base64 string.
    """

    name: Optional[Text] = None
    image: Optional[Text] = None


@dataclasses.dataclass
class GraphicsCollection(BaseModelCardField):
    """A collection of graphics.

    Each ```graphic``` in the ```collection``` field has both a ```name``` and
    an ```image```. For instance, you might want to display a graph showing the
    number of examples belonging to each class in your training dataset:

    ```python

    model_card.model_parameters.data.train.graphics.collection = [
      {'name': 'Training Set Size', 'image': training_set_size_barchart},
    ]
    ```

    Then, provide a description of the graph:

    ```python

    model_card.model_parameters.data.train.graphics.description = (
      'This graph displays the number of examples belonging to each class ',
      'in the training dataset. ')
    ```

    Attributes:
      description: The description of graphics.
      collection: A collection of graphics.
    """

    description: Optional[Text] = None
    collection: List[Graphic] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SensitiveData(BaseModelCardField):
    """Sensitive data, such as PII (personally-identifiable information).

    Attributes:
      sensitive_data: A description of any sensitive data that may be present in a
        dataset. Be sure to note PII information such as names, addresses, phone
        numbers, etc. Preferably, such info should be scrubbed from a dataset if
        possible. Note that even non-identifying information, such as zip code,
        age, race, and gender, can be used to identify individuals when
        aggregated. Please describe any such fields here.
    """

    sensitive_data: List[Text] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Dataset(BaseModelCardField):
    """Provide some information about a dataset used to generate a model.

    Attributes:
      name: The name of the dataset.
      description: The description of dataset.
      link: A link to the dataset.
      sensitive: Does this dataset contain human or other sensitive data?
      graphics: Visualizations of the dataset.
    """

    name: Optional[Text] = None
    description: Optional[Text] = None
    link: Optional[Text] = None
    sensitive: Optional[SensitiveData] = dataclasses.field(
        default_factory=SensitiveData
    )
    graphics: GraphicsCollection = dataclasses.field(
        default_factory=GraphicsCollection
    )


@dataclasses.dataclass
class ModelParameters(BaseModelCardField):
    """Parameters for construction of the model.

    Attributes:
      model_architecture: specifies the architecture of your model.
      data: specifies the datasets used to train and evaluate your model.
      input_format: describes the data format for inputs to your model.
      output_format: describes the data format for outputs from your model.
    """

    model_architecture: Optional[Text] = None
    data: List[Dataset] = dataclasses.field(default_factory=list)
    input_format: Optional[Text] = None
    output_format: Optional[Text] = None


@dataclasses.dataclass
class PerformanceMetric(BaseModelCardField):
    """The details of the performance metric.

    Attributes:
      type: What performance metric are you reporting on?
      value: What is the value of this performance metric?
      slice: What slice of your data was this metric computed on?
    """

    # TODO(b/179415408): add fields (name, value, confidence_interval,
    # threshold, slice) after gathering requirements
    # (potential clients: Jigsaw)
    # The following fields are EXPERIMENTAL and introduced for migration
    # purpose.
    type: Optional[Text] = None
    value: Optional[Text] = None
    slice: Optional[Text] = None


@dataclasses.dataclass
class QuantitativeAnalysis(BaseModelCardField):
    """The quantitative analysis of a model.

    Identify relevant performance metrics and display values. Let’s say you’re
    interested in displaying the accuracy and false positive rate (FPR) of a
    cat vs. dog classification model. Assuming you have already computed both
    metrics, both overall and per-class, you can specify metrics like so:

    ```python
    model_card.quantitative_analysis.performance_metrics = [
      {'type': 'accuracy', 'value': computed_accuracy},
      {'type': 'accuracy', 'value': cat_accuracy, 'slice': 'cat'},
      {'type': 'accuracy', 'value': dog_accuracy, 'slice': 'dog'},
      {'type': 'fpr', 'value': computed_fpr},
      {'type': 'fpr', 'value': cat_fpr, 'slice': 'cat'},
      {'type': 'fpr', 'value': dog_fpr, 'slice': 'dog'},
    ]
    ```

    Attributes:
      performance_metrics: The performance metrics being reported.
      graphics: A collection of visualizations of model performance.
    """

    performance_metrics: List[PerformanceMetric] = dataclasses.field(
        default_factory=list
    )
    graphics: GraphicsCollection = dataclasses.field(
        default_factory=GraphicsCollection
    )


@dataclasses.dataclass
class User(BaseModelCardField):
    """A type of user for a model.

    Attributes:
      description: A description of a user.
    """

    description: Optional[Text] = None


@dataclasses.dataclass
class UseCase(BaseModelCardField):
    """A type of use case for a model.

    Attributes:
      description: A description of a use case.
    """

    description: Optional[Text] = None


@dataclasses.dataclass
class Limitation(BaseModelCardField):
    """A limitation a model.

    Attributes:
      description: A description of the limitation.
    """

    description: Optional[Text] = None


@dataclasses.dataclass
class Tradeoff(BaseModelCardField):
    """A tradeoff for a model.

    Attributes:
      description: A description of the tradeoff.
    """

    description: Optional[Text] = None


@dataclasses.dataclass
class Risk(BaseModelCardField):
    """Information about risks involved when using the model.

    Attributes:
      name: The name of the risk.
      mitigation_strategy: A mitigation strategy that you've implemented, or one
        that you suggest to users.
    """

    name: Optional[Text] = None
    mitigation_strategy: Optional[Text] = None


@dataclasses.dataclass
class Considerations(BaseModelCardField):
    """Considerations related to model construction, training, and application.

    The considerations section includes qualitative information about your
    model, including some analysis of its risks and limitations. As such, this
    section usually requires careful consideration, and conversations with many
    relevant stakeholders, including other model developers, dataset producers,
    and downstream users likely to interact with your model, or be affected by
    its outputs.

    Attributes:
      users: Who are the intended users of the model? This may include
        researchers, developers, and/or clients. You might also include
        information about the downstream users you expect to interact with your
        model.
      use_cases: What are the intended use cases of the model? What use cases
        are out-of-scope?
      limitations: What are the known limitations of the model? This may
        include technical limitations, or conditions that may degrade model
        performance.
      tradeoffs: What are the known accuracy/performance tradeoffs for the
      model?
      ethical_considerations: What are the ethical risks involved in
        application of this model? For each risk, you may also provide a
        mitigation strategy that you've implemented, or one that you suggest to
        users.
    """

    users: List[User] = dataclasses.field(default_factory=list)
    use_cases: List[UseCase] = dataclasses.field(default_factory=list)
    limitations: List[Limitation] = dataclasses.field(default_factory=list)
    tradeoffs: List[Tradeoff] = dataclasses.field(default_factory=list)
    ethical_considerations: List[Risk] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass
class ModelCard(BaseModelCardField):
    """Fields used to generate the Model Card.

    Attributes:
      model_details: Descriptive metadata for the model.
      model_parameters: Technical metadata for the model.
      quantitative_analysis: Quantitative analysis of model performance.
      considerations: Any considerations related to model construction,
        training, and application.
    """

    model_details: ModelDetails = dataclasses.field(
        default_factory=ModelDetails
    )
    model_parameters: ModelParameters = dataclasses.field(
        default_factory=ModelParameters
    )
    quantitative_analysis: QuantitativeAnalysis = dataclasses.field(
        default_factory=QuantitativeAnalysis
    )
    considerations: Considerations = dataclasses.field(
        default_factory=Considerations
    )

    def to_json(self) -> Text:
        """Write ModelCard to JSON."""
        model_card_dict = self.to_dict()
        model_card_dict[
            _SCHEMA_VERSION_STRING
        ] = get_latest_schema_version()
        validate_json_schema(model_card_dict)
        return json_lib.dumps(model_card_dict, indent=2)

    def _from_json(self, json_dict: Dict[Text, Any]) -> "ModelCard":
        """Read ModelCard from JSON.

        If ModelCard fields have already been set, this function will overwrite
        any existing values.

        WARNING: This method's interface may change in the future, do not use
        for critical workflows.

        Args:
          json_dict: A JSON dict from which to populate fields in the model
            card schema.

        Returns:
          self

        Raises:
          JSONDecodeError: If `json_dict` is not a valid JSON string.
          ValidationError: If `json_dict` does not follow the model card JSON
            schema.
          ValueError: If `json_dict` contains a value not in the class or
            schema definition.
        """

        def _populate_from_json(
            json_dict: Dict[Text, Any], field: BaseModelCardField
        ) -> BaseModelCardField:
            for subfield_key in json_dict:
                if subfield_key.startswith(_SCHEMA_VERSION_STRING):
                    continue
                elif not hasattr(field, subfield_key):
                    raise ValueError(
                        "BaseModelCardField %s has no such field named '%s.'"
                        % (field, subfield_key)
                    )
                elif isinstance(json_dict[subfield_key], dict):
                    subfield_value = _populate_from_json(
                        json_dict[subfield_key], getattr(field, subfield_key)
                    )
                elif isinstance(json_dict[subfield_key], list):
                    subfield_value = []
                    for item in json_dict[subfield_key]:
                        if isinstance(item, dict):
                            new_object = field.__annotations__[
                                subfield_key
                            ].__args__[
                                0
                            ]()  # pytype: disable=attribute-error
                            subfield_value.append(
                                _populate_from_json(item, new_object)
                            )
                        else:  # if primitive
                            subfield_value.append(item)
                else:
                    subfield_value = json_dict[subfield_key]
                setattr(field, subfield_key, subfield_value)
            return field

        validate_json_schema(json_dict)
        self.clear()
        _populate_from_json(json_dict, self)
        return self
