import sys
import os
import json
import ollama

# Add the path to the benchmark module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from benchmark.model_class import ModelPrediction, Model

from benchmark.benchmark import Benchmark


class OllamaModel(Model):
    def __init__(self, model: str = "deepseek-r1:8b"):
        super().__init__()
        self.model = model

    def get_prediction_metrics(self):
        # _metrics = ["error_type", "severity", "description", "solution"]
        _metrics = ["error_type", "severity"]
        return _metrics

    def system_prompt(self) -> str:
        system_prompt = f"""
        You are a log analysis assistant. Your task is to read production log lines and extract four fields:
        1. error_type: One of ["no_error", "warning", "runtime", "fatal"].
        2. severity: One of ["notice", "warn", "error"].
        3. description: A one-line description of the log.
        4. solution: A one-line proposed solution if the log is an error or warning; if not applicable, leave it empty.

        All responses must be in JSON format with exactly these keys and no additional text. Do not include any internal reasoning or chain-of-thought in your final output.
        """
        response = ollama.chat(model=self.model, messages=[{"role": "system", "content": system_prompt}])

        return response.get("message", {}).get("content", "")

    def predict(self, text: str) -> ModelPrediction:
        prompt = f"""
        Analyze the following production log line and extract the required details according to the instructions above.

        Log line: "{text}"

        Here are a few examples of expected output:
        Example 1:
        Input: [2898323] AH00163: Apache/2.4.52 (Ubuntu) OpenSSL/3.0.2 configured -- resuming normal operations]
        Output: {{"error_type": "no_error", "severity": "notice", "description": "The log line indicates that the Apache server has successfully completed its configuration and is resuming normal operations.", "solution": "No action is required as this is a normal operational message. The server is functioning correctly."}}

        Example 2:
        Input: [Creating 32 session mutexes based on 150 max processes and 0 max threads.]
        Output: {{"error_type": "no_error", "severity": "notice", "description": "The log line indicates the creation of session mutexes, which is a normal part of system initialization. The numbers are based on configuration parameters for maximum processes and threads.", "solution": "No action is required as this is a standard informational message."}}

        Example 3:
        Input: [ok /etc/httpd/conf/workers2.properties]
        Output: {{"error_type": "no_error", "severity": "notice", "description": "The log indicates that the file /etc/httpd/conf/workers2.properties was successfully checked and no issues were found.", "solution": "No specific action is required since the file check was successful. However, it's a good practice to periodically review configuration files for any changes or potential issues."}}

        Example 4:
        Input: [3623857] [client 150.136.69.140:58487] PHP Warning:  Undefined array key 'HTTP_USER_AGENT' in /var/www/sylvainkalache.com/wp-content/themes/themify-base/themify/themify-functions.php on line 494]
        Output: {{"error_type": "warning", "severity": "warn", "description": "The logged line indicates a PHP warning where the code attempted to access an undefined array key 'HTTP_USER_AGENT'. This typically occurs when the client request does not include the User-Agent header, leading to the warning.", "solution": "To resolve this issue, check if the 'HTTP_USER_AGENT' key exists in the $_SERVER array before accessing it. Use isset() or array_key_exists() functions to prevent the warning."}}

        Example 5:
        Input: [3466297] [client 46.101.103.154:1901] PHP Warning:  Undefined array key 'host' in /var/www/rootly.com/wp-includes/canonical.php on line 717]
        Output: {{"error_type": "warning", "severity": "warn", "description": "The log indicates a PHP warning where an undefined array key 'host' is being accessed. This typically occurs when the code attempts to access an array element that does not exist.", "solution": "Check the code in canonical.php around line 717 to ensure the 'host' key is properly set before accessing it. Consider using isset() or array_key_exists() to validate the key's existence."}}

        Example 6:
        Input: [219.133.247.171] Directory index forbidden by rule: /var/www/html/]
        Output: {{"error_type": "warning", "severity": "error", "description": "The log indicates that access to the directory index at /var/www/html/ has been blocked due to a security rule, preventing directory listing.", "solution": "Check if directory indexing should be allowed. If not needed, ensure Options -Indexes is set in your server configuration (e.g., .htaccess or apache.conf). Verify file and directory permissions are correctly configured."}}

        Example 7:
        Input: [child init 1 -2]
        Output: {{"error_type": "runtime", "severity": "error", "description": "The log line indicates an error during child process initialization with a return code of -2, suggesting a runtime issue.", "solution": "Check the application's documentation for the meaning of return code -2. Verify environment setup and dependencies required for proper initialization."}}

        Example 8:
        Input: [212.238.198.203] script not found or unable to stat: /var/www/cgi-bin/]
        Output: {{"error_type": "runtime", "severity": "error", "description": "The system attempted to access the script at /var/www/cgi-bin/openwebmail but was unable to find it or determine its status. This indicates a potential issue with the script's existence, permissions, or configuration.", "solution": "Check if the script exists at the specified path and verify its permissions. Ensure that the script is executable by the appropriate user or group. If the script does not exist, install or restore it as needed."}}

        Example 9:
        Input: [3434944] [client 143.110.217.244:59516] AH01630: client denied by server configuration: /var/www/rootly.com/server-status]
        Output: {{"error_type": "runtime", "severity": "error", "description": "The log indicates that a client was denied access to the server-status resource due to server configuration restrictions.", "solution": "Check the server's access control lists and ensure proper permissions are set for accessing /server-status. Verify if the client IP should have access and adjust configurations accordingly."}}

        Example 10:
        Input: [3462505] [client 128.199.178.241:59141] PHP Fatal error:  Uncaught Error: Call to undefined function _x() in /var/www/rootly.com/wp-includes/block-patterns/query-grid-posts.php:9\nStack trace:\n#0 {{main}}\n  thrown in /var/www/rootly.com/wp-includes/block-patterns/query-grid-posts.php on line 9, referer: www.google.com]
        Output: {{"error_type": "fatal", "severity": "error", "description": "A fatal error occurred due to an undefined function call to _x(). The error is located in the file query-grid-posts.php on line 9.", "solution": "Check if the function _x() is properly defined and included. Ensure all necessary files are correctly referenced."}}

        Example 11:
        Input: [3462412] [client 128.199.178.241:64722] PHP Fatal error:  Uncaught Error: Class 'WP_REST_Meta_Fields' not found in /var/www/rootly.com/wp-includes/rest-api/fields/class-wp-rest-term-meta-fields.php:17\nStack trace:\n#0 {{main}}\n  thrown in /var/www/rootly.com/wp-includes/rest-api/fields/class-wp-rest-term-meta-fields.php on line 17, referer: www.google.com]
        Output: {{"error_type": "fatal", "severity": "error", "description": "The error indicates that the PHP class 'WP_REST_Meta_Fields' is missing, which is required for proper functionality. This typically occurs when WordPress core files are corrupted or outdated.", "solution": "Verify that all WordPress core files are up to date and correctly installed. Reinstalling WordPress core files may resolve this issue."}}

        Example 12:
        Input: [3477269] [client 204.10.194.48:53404] PHP Fatal error:  Uncaught Error: Class 'IXR_Client' not found in /var/www/sylvainkalache.com/wp-includes/IXR/class-IXR-clientmulticall.php:8\nStack trace:\n#0 {{main}}\n  thrown in /var/www/sylvainkalache.com/wp-includes/IXR/class-IXR-clientmulticall.php on line 8]
        Output: {{"error_type": "fatal", "severity": "error", "description": "The error indicates that the IXR_Client class is missing, which is required for XML-RPC functionality in WordPress. This could be due to a missing or corrupted file in the wp-includes/IXR directory.", "solution": "Verify that all core WordPress files are present and correctly installed. Reinstalling WordPress core files or specifically the IXR library might resolve this issue."}}

        Return only a JSON object with the keys "error_type", "severity", "description", "solution".
        """
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        # Extract the response content.
        response_text = response.get("message", {}).get("content", "")
        # print("Raw response:")
        # print(response_text)

        # Get the content after </think> tag
        response_text = response_text.split("</think>")[1].strip()
        # Get the content between {}, including the curly braces
        response_text = response_text[response_text.find("{") : response_text.rfind("}") + 1].strip()

        try:
            parsed = json.loads(response_text)
            output = ModelPrediction(
                input=text,
                error_type=parsed.get("error_type", None),
                severity=parsed.get("severity", None),
                description=parsed.get("description", None),
                solution=parsed.get("solution", None),
            )
        except json.JSONDecodeError:
            output = ModelPrediction(
                input=text,
                error_type="no_error",
                severity="notice",
                description="",
                solution="",
            )

        return output


if __name__ == "__main__":

    model = "deepseek-r1:70b"

    # actual data
    data_path = "data/validation/actual_validation.csv"

    deepseek_r1 = OllamaModel(model=model)
    deepseek_r1.system_prompt()

    benchmark = Benchmark(model=deepseek_r1, dataset_path=data_path, delimiter="|")
    losses, predictions, data_length, skipped_examples = benchmark.run_benchmark()

    print(f"Losses: {losses}")
    print(f"Total data length: {data_length}")
    print(f"Skipped examples: {skipped_examples}")

    # Write predictions to a file
    with open("predictions_actual.json", "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction.to_dict()))
            f.write("\n")

    # synthetic data
    data_path = "data/validation/synthetic_validation.csv"

    deepseek_r1 = OllamaModel(model=model)
    deepseek_r1.system_prompt()

    benchmark = Benchmark(model=deepseek_r1, dataset_path=data_path, delimiter="|")
    losses, predictions, data_length, skipped_examples = benchmark.run_benchmark()

    print(f"Losses: {losses}")
    print(f"Total data length: {data_length}")
    print(f"Skipped examples: {skipped_examples}")

    # Write predictions to a file
    with open("predictions_synthetic.json", "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction.to_dict()))
            f.write("\n")
