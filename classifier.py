"""
Classification interface for the one-shot AI assessment of a moving
image crop.

Currently: MockClassifier, which works without a real model (purely
rule-based, e.g. by area/aspect ratio) so the whole app already runs
end-to-end with simulated images.

Later: RknnClassifier with the same interface (classify(crop)
-> ClassificationResult), using rknn-toolkit-lite2 internally to run
the .rknn model converted for the RV1106. The rest of the app must
not change for that.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    is_animal: bool


class Classifier(ABC):
    @abstractmethod
    def classify(self, crop) -> ClassificationResult:
        raise NotImplementedError


class MockClassifier(Classifier):
    """
    Placeholder classifier for development without a trained model.
    Classifies purely by crop size, so the rest of the app (tracking,
    display) can be tested end-to-end.
    """

    def __init__(self, min_area_for_animal=1500):
        self.min_area_for_animal = min_area_for_animal

    def classify(self, crop) -> ClassificationResult:
        h, w = crop.shape[:2]
        area = h * w
        is_animal = area >= self.min_area_for_animal
        confidence = min(0.99, area / (self.min_area_for_animal * 3))
        label = "animal" if is_animal else "not_animal"
        return ClassificationResult(label=label, confidence=round(confidence, 2), is_animal=is_animal)


class MockCatClassifier(Classifier):
    """
    Placeholder cat classifier for development without a trained model: it
    only looks at the crop size, so the whole perception stack can be
    exercised.

    ``is_animal`` means "the requested target class was found". The labels
    are English ("cat"/"not_cat") because the target detector matches the
    label against its target labels.
    """

    def __init__(self, min_area_for_cat=1500, label="cat", negative_label="not_cat"):
        self.min_area_for_cat = min_area_for_cat
        self.label = label
        self.negative_label = negative_label

    def classify(self, crop) -> ClassificationResult:
        h, w = crop.shape[:2]
        area = h * w
        is_cat = area >= self.min_area_for_cat
        confidence = min(0.99, area / (self.min_area_for_cat * 3))
        return ClassificationResult(
            label=self.label if is_cat else self.negative_label,
            confidence=round(confidence, 2),
            is_animal=is_cat,
        )


class RknnClassifier(Classifier):
    """
    Placeholder for the later NPU classification on the RV1106.
    Not implemented yet - the structure shows how the swap will look
    once a .rknn model is available.
    """

    def __init__(self, model_path: str, labels: list[str]):
        self.model_path = model_path
        self.labels = labels
        raise NotImplementedError(
            "RknnClassifier will be implemented once a trained model "
            "converted with RKNN-Toolkit2 is available."
        )

    def classify(self, crop) -> ClassificationResult:
        raise NotImplementedError
