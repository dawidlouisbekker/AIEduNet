from flask import Blueprint, render_template, send_file, request, send_from_directory

views = Blueprint('views', __name__)



@views.route('/')
def index():
    return render_template('index.html')

@views.route('/download-zip', methods=['POST'])
def download_zip():
    json_response = request.get_json()
    os_version = json_response.get('os_version')
    with open('json.txt', 'w') as f:
        f.write(os_version)
    return send_file('download/windows/file.zip', as_attachment=True, attachment_filename='file.zip',mimetype='application/zip'), send_from_directory('download/windows', 'file.zip', as_attachment=True)

#open file of connected devices correlated with os. Have to use flask


