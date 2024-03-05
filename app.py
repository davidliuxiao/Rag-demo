from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json['message']
    # response = openai.Completion.create(
    #     engine="text-davinci-003",
    #     prompt=user_message,
    #     max_tokens=150
    # )

    # chatgpt_response = response.choices[0].text.strip()
    # return {'message': chatgpt_response}
    return {'message': "hello"}


if __name__ == '__main__':
    app.run(debug=True)