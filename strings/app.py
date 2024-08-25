from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def add(s):
    l = []
    negative_numbers = []
    i = 0
    while i < len(s):
        if s[i] == '\n':
            i += 1
            continue
        if s[i] == '-':
            # Start capturing the negative number
            if i + 1 < len(s) and s[i + 1].isdigit():
                neg_number = '-'  # Initialize with the negative sign
                i += 1
                while i < len(s) and s[i].isdigit():
                    neg_number += s[i]
                    i += 1
                negative_numbers.append(int(neg_number))
                continue
            else:
                return "not allowed"
        
        try:
            if s[i].isdigit():
                l.append(int(s[i]))
        except ValueError:
            continue
        
        i += 1
    
    if negative_numbers:
        return f"negative numbers not allowed: {', '.join(map(str, negative_numbers))}"
    
    return sum(l)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    input_string = data.get('s', '')
    result = add(input_string)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
