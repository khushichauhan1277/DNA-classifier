from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# DNA Training Dataset
sequences = [
    "ATGCGATC",
    "ATGCGTAC",
    "GCGCGTAT",
    "GCGCGCGC"
]

labels = [
    "Group A",
    "Group A",
    "Group B",
    "Group B"
]

# DNA Feature Extraction
def extract_features(dna):

    a = dna.count("A")
    t = dna.count("T")
    g = dna.count("G")
    c = dna.count("C")

    length = len(dna)

    gc = ((g + c) / length) * 100

    return [a, t, g, c, length, gc]


# Prepare Training Data
X = [extract_features(s) for s in sequences]
y = labels

# Train Machine Learning Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


# Flask Website
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    dna = ""
    features = None
    error = None

    if request.method == "POST":

        dna = request.form.get("sequence", "").upper().strip()
        dna = "".join(dna.split())

        if not dna:
            error = "Please enter a DNA sequence."

        elif any(base not in "ATGC" for base in dna):
            error = "Only A, T, G and C are allowed."

        else:
            features = extract_features(dna)

            prediction = model.predict([features])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        dna=dna,
        features=features,
        error=error
    )


# Run Website
if __name__ == "__main__":
    app.run(debug=True)