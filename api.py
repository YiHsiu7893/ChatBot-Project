from flask import Flask, request, render_template, jsonify
import Chatbot

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"


@app.route('/main', methods=['GET'])
def show_main():
    return render_template('main.html')


@app.route('/func', methods=['POST'])
def run_chatbot():
    if request.method == 'POST':
        input = request.get_json()
        text = ""
        for key in input:
            text = text + input[key]
        # text = request.values['user_input']
        output = Chatbot.main(text)
        return jsonify(output)
    

@app.route('/drugs', methods=['POST'])
def run_otc():
    if request.method == 'POST':
        json = request.get_json()
        option = json['option'] # yes or no
        input = json['user_input']
        lang = json['lang']
        if option == 'yes':
            result = Chatbot.otc_recmd(input, lang).split('\n')
            
            drugs = dict()
            for i in range(len(result)):
                if result[i] != '':
                    drugs[i+1] = result[i]
            return jsonify(drugs)
        else:
            return ''
    

if __name__ == '__main__':
    # Chatbot.update_otc()
    app.debug = True
    app.run()
