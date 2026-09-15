"""
Klassifikations-Schnittstelle für die einmalige KI-Bewertung eines
bewegten Bildausschnitts.

Aktuell: MockClassifier, der ohne echtes Modell arbeitet (rein
regelbasiert, z.B. nach Fläche/Seitenverhältnis), damit die
gesamte Pipeline schon jetzt end-to-end mit simulierten Bildern
lauffähig ist.

Später: RknnClassifier mit derselben Schnittstelle (classify(crop)
-> ClassificationResult), die intern rknn-toolkit-lite2 nutzt, um
das auf dem RV1106 konvertierte .rknn-Modell auszuführen. Der Rest
der Pipeline (pipeline.py) muss dafür nicht verändert werden.
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
    Platzhalter-Klassifikator für die Entwicklung ohne trainiertes Modell.
    Klassifiziert rein nach Bildausschnitt-Größe, damit der Rest der
    Pipeline (Tracking, Anzeige) end-to-end getestet werden kann.
    """

    def __init__(self, min_area_for_animal=1500):
        self.min_area_for_animal = min_area_for_animal

    def classify(self, crop) -> ClassificationResult:
        h, w = crop.shape[:2]
        area = h * w
        is_animal = area >= self.min_area_for_animal
        confidence = min(0.99, area / (self.min_area_for_animal * 3))
        label = "tier" if is_animal else "kein_tier"
        return ClassificationResult(label=label, confidence=round(confidence, 2), is_animal=is_animal)


class RknnClassifier(Classifier):
    """
    Platzhalter für die spätere NPU-Klassifikation auf dem RV1106.
    Noch nicht implementiert - Struktur zeigt, wie der Austausch
    später aussehen wird, sobald ein .rknn-Modell vorliegt.
    """

    def __init__(self, model_path: str, labels: list[str]):
        self.model_path = model_path
        self.labels = labels
        raise NotImplementedError(
            "RknnClassifier wird implementiert, sobald ein trainiertes "
            "und mit RKNN-Toolkit2 konvertiertes Modell vorliegt."
        )

    def classify(self, crop) -> ClassificationResult:
        raise NotImplementedError
