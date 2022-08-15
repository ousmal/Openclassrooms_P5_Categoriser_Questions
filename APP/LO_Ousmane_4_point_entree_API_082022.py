from flask import Flask, render_template, request
import joblib

app = Flask(__name__, template_folder='templates', static_folder='templates/static')

keyword_model = joblib.load("C:/Users/ousma/PycharmProjects/Project5/model_pipeline.pkl")
transformer = joblib.load("C:/Users/ousma/PycharmProjects/Project5/mlb_transformer.pkl")


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    title = request.form.get('title')
    print(title)
    body = request.form.get('body')
    print(body)

    print("String format required for Machine Learning prediction")
    post = title + " " + body
    post = [post]
    keyword = keyword_model.predict(post)
    keyword = keyword
    keyword = transformer.inverse_transform(keyword)
    keyword = [x for x in keyword if x != ()]
    keyword = list(set(keyword))
    return render_template('predict.html',
                           title=title,
                           body=body,
                           prediction_text="Tags suggested{}".format(keyword))


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)

