from flask import Flask, jsonify

mcp_server = Flask(__name__)

@mcp_server.route('/status')
def status():
    return jsonify({
        "server": "running",
        "purpose": "IT Transition Processor"
    })

if __name__ == "__main__":
    mcp_server.run(port=5001, debug=True)
