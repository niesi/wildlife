# Wildlife-Erkennungspipeline (Prototyp)

Grundgerüst für Bewegungserkennung -> einmalige KI-Klassifikation ->
Tracking, entwickelbar am PC mit simulierten Bildern, bevor die
RV1106-Kamera-Hardware verfügbar ist.

## Installation

```bash
pip install opencv-python numpy --break-system-packages
```

## Starten

Mit rein synthetischen Testbildern (kein Video/Kamera nötig):
```bash
python pipeline.py --source synthetic
```

Die synthetische Quelle verwendet standardmäßig eine einfache Tier-Silhouette.
Ein eigenes Bild kann als Sprite verwendet werden (für saubere Kanten am besten
ein PNG mit transparentem Hintergrund):
```bash
python pipeline.py --source synthetic --sprite pfad/zum/tier.png
```
Das Sprite wird gespiegelt, gedreht und abhängig von seiner Höhe im Bild
skaliert. Es kann den Bildrand vollständig verlassen. Der Hintergrund verändert
Helligkeit und Farbe langsam in mehreren Bereichen und durchläuft einen
Tages-/Nachtzyklus.

Mit einer Videodatei (z.B. heruntergeladener Wildlife-Kamerafallen-Clip):
```bash
python pipeline.py --source video_file --path pfad/zum/clip.mp4
```

Die Videowiedergabe verwendet die FPS-Metadaten der Datei und berücksichtigt
Dekodierung und Verarbeitung bei der Wartezeit. Bei ungültigen FPS-Metadaten
werden 30 FPS verwendet. Ist die Verarbeitung langsamer als das Frameintervall,
läuft die Wiedergabe langsamer; Frames werden nicht übersprungen. Die Zeitsteuerung
ist durch die GUI-/Betriebssystem-Timer begrenzt und verwendet bei Videos mit
variabler Framerate die gemeldete nominale Framerate.

Bewegungsbereiche werden zwischen `--min-area` und `--max-area` akzeptiert.
Große Hintergrundänderungen oberhalb von `--max-area` werden nicht getrackt.

Mit der Webcam am PC:
```bash
python pipeline.py --source webcam --index 0
```

Alle Bildquellen liefern Graustufenbilder. Bewegungserkennung, Klassifikation
und Tracking arbeiten ausschließlich auf unveränderten Graustufenbildern.
Nur die Anzeige verwendet eine separate BGR-Kopie: Boxen und Beschriftungen
bleiben farbig; `--show-motion-mask` zeigt Bewegungspixel in Rot.

Unter dem Video stehen drei getrennte Statuszeilen:
- **MOTION (blau):** Anzahl der gefilterten Bewegungsregionen oder der Hinweis,
  dass die Erkennung während des Trackings übersprungen wurde. Boxen: `MOTION #n`.
- **TRACKER (gelb):** idle, started, tracking, lost oder nach Ablehnung gestoppt.
  Boxen: `TRACKER: label`.
- **CLASSIFICATION (MOCK, grün/rot):** letztes Label, Konfidenz und Annahme/Ablehnung.
  Frische Ergebnisse zeigen die Anzahl klassifizierter Ausschnitte; ältere Ergebnisse
  sind ausdrücklich mit ihrem Alter in Frames markiert. `CLASSIFY`-Boxen zeigen die
  tatsächlich klassifizierten Ausschnitte nur auf dem jeweiligen Frame.

Der Klassifikator bleibt ein regelbasierter Mock, kein trainiertes KI-Modell.
Die Anzeige löst keine zusätzlichen Klassifikationen oder Bewegungserkennungen aus.

Taste `q` beendet die Anzeige.

## Struktur

- `frame_source.py` – austauschbare Bildquelle (Datei/Webcam/synthetisch,
  später CSI-Kamera über dieselbe Schnittstelle).
- `motion_detector.py` – MOG2-Bewegungserkennung, liefert Bounding-Boxes.
- `classifier.py` – Klassifikations-Schnittstelle. `MockClassifier` ist
  ein Platzhalter ohne echtes Modell. `RknnClassifier` ist als Gerüst für
  die spätere NPU-Klassifikation vorbereitet (RKNN-Toolkit2).
- `tracker.py` – KCF-Tracking nach bestätigter Klassifikation, mit
  periodischer Re-Klassifikation.
- `pipeline.py` – verbindet alles, Kommandozeilen-Einstiegspunkt.
- `tests/` – Unit-Tests für Pipeline und Turm, gestartet mit
  `python -m unittest discover` aus dem Projektverzeichnis.

## Nächste Schritte

1. Echte Wildlife-Videos/Bilder als Testmaterial einbinden
   (z.B. LILA-BC-Datensatz) statt nur `synthetic`.
2. Ein kleines Klassifikationsmodell trainieren, `MockClassifier`
   durch ein echtes (zunächst noch PC-seitiges TensorFlow/TFLite-
   Modell) ersetzen.
3. Sobald ein `.rknn`-Modell vorliegt: `RknnClassifier` implementieren,
   `pipeline.py` unverändert lassen (nur den Classifier austauschen).
4. Sobald die CSI-Kamera verfügbar ist: `CsiFrameSource` in
   `frame_source.py` ergänzen, restliche Pipeline bleibt unverändert.

## Water turret (cat deterrent)

The repository also contains the software structure for an automatic water
turret that aims at cats in the garden. The decision layer is implemented and
unit tested; camera, tracking, servo and pump wiring follow in the next phase.

- `state_machine.py` – states, events, transitions and state timeouts.
- `safety.py` – arming, spray zone, cooldown, spray budgets, time window,
  emergency stop.
- `turret_config.py` – all parameters as dataclasses, readable and writable as
  UTF-8 JSON.
- `tests/test_state_machine.py`, `tests/test_safety.py`, `tests/test_turret_config.py` – 87 tests.
- `TURRET.md` – architecture, state diagram, transition table, safety rules and
  the planned interfaces of the next phase.

```bash
python state_machine.py        # prints the transition and timeout tables
python -m unittest discover -v # pipeline tests + turret tests
```

The turret is disarmed by default: it neither aims nor sprays until it is armed
on purpose.
