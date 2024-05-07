from flask import Flask, request, render_template
import Chatbot

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

@app.route('/main', methods=['GET'])
def show_main():
    return render_template('main.html')


@app.route('/func', methods=['POST'])
def run_chatbot():
    if request.method == 'POST':
        text = request.values['user_input']
        output = Chatbot.main(text)
        return output
    

@app.route('/drugs', methods=['POST'])
def run_otc():
    if request.method == 'POST':
        option = request.values['option'] # yes or no
        input = request.values['user_input']
        if option == 'yes':
            result = Chatbot.otc_recmd(input).split('\n')

            drugs = ""
            for i in range(len(result)):
                drugs += "<li>" + result[i] + "</li>"
            return drugs
        else:
            return ''
    

if __name__ == '__main__':
    app.debug = True
    app.run()
