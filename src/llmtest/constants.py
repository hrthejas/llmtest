import os
from environs import Env

env = Env()

MAX_NEW_TOKENS = env.int("MAX_NEW_TOKENS", 800)
DOCS_BASE_PATH = env.str("DOCS_BASE_PATH", "/content/drive/Shareddrives/Engineering/Chatbot/thejas/data/")
INDEX_BASE_PATH = env.str("INDEX_BASE_PATH", "/content/drive/Shareddrives/Engineering/Chatbot/thejas/indexes/")
HF_INDEX_BASE_PATH = env.str("HF_INDEX_BASE_PATH", "/content/drive/Shareddrives/Engineering/Chatbot/thejas/indexes/hf/")
OAI_INDEX_BASE_PATH = env.str("OAI_INDEX_BASE_PATH",
                              "/content/drive/Shareddrives/Engineering/Chatbot/thejas/indexes/openai/")

DOC_INDEX_NAME_PREFIX = env.list("DOC_INDEX_NAME_PREFIX", ["doc_index"])
API_INDEX_NAME_PREFIX = env.list("API_INDEX_NAME_PREFIX", ["api_index_csv"])
DEFAULT_MODEL_NAME = env.str("DEFAULT_MODEL_NAME", "thr10/thr-wlm-15b-3gb")
USE_4_BIT_QUANTIZATION = env.bool("USE_4_BIT_QUANTIZATION", True)
SET_DEVICE_MAP = env.bool("SET_DEVICE_MAP", True)

GDRIVE_MOUNT_BASE_PATH = env.str("GDRIVE_MOUNT_BASE_PATH", "/content/drive")

OPEN_AI_TEMP = env.int("OPEN_AI_TEMP", 0)

OPEN_AI_MODEL_NAME = env.str("OPEN_AI_MODEL_NAME", "gpt-3.5-turbo-16k")

USER_NAME = env.str("USER_NAME", "user@infoworks.io")

DEFAULT_DEVICE_MAP = env.str("DEFAULT_DEVICE_MAP", "auto")

MYSQL_HOST = env.str("MYSQL_HOST", "35.224.111.132")
MYSQL_USER = env.str("MYSQL_USER", "infoworks")
MYSQL_PASSWD = env.str("MYSQL_PASSWD", "IN11**rk")
MYSQL_DB = env.str("MYSQL_DB", "generative_ai")

DEFAULT_PROMPT_FOR_DOC = """Below is context and instruction use the context and write a response that appropriately completes the request.

###CONTEXT: 
{context}
=========

###INSTRUCTION:
{question} 

"""

DEFAULT_CSV_PARSE_ARGS = {
    "delimiter": ",",
    "fieldnames": ["Method", "Path", "Operation", "Description", "Query Parameters", "Request Parameters"],
}

CSV_DOC_PARSE_ARGS = env.dict("CSV_DOC_PARSE_ARGS", DEFAULT_CSV_PARSE_ARGS)

CSV_DOC_EMBEDDING_SOURCE_COLUMN = env.str("CSV_DOC_EMBEDDING_SOURCE_COLUMN", "Description")

DEFAULT_PROMPT_FOR_API_OLD = """Below is an instruction that describes a task. write a response that appropriately completes the request.

###INSTRUCTION:

You are a REST API assistant working at Infoworks, but you are also an expert programmer.
You are to complete the user request by composing a series of commands.
Use the minimum number of commands required.

IMPORTANT - Strictly follow below conditions while generating output.
1. Look for any value for the query or body parameters in the prompt before generating commands.
2. From the context provided below 'Request parameter' will give pipe delimited body parameters use that for Input Command.
3. From the context provided below 'Query parameter' will give pipe delimited query parameters use that for Input Command.
4. From the context provided below 'Method' will give you api call method GET/POST/PATCH.
5. For every POST and PATCH request ask user input for body parameters.


IMPORTANT - The commands you have available are:

| Command | Arguments  | Description                              |
| ------- | ---------  | ---------------------------------------- |
| Input   | question   | Ask input from user                      |
| execute | APIRequest | execute an Infoworks v3 REST API request |

IMPORTANT - Use these commands to Output the commands in JSON as an abstract syntax tree in one of the below format depending on <Method>:

[
  //Ask this for every parameter
  {{
    "command": "input",
    "arguments": "Enter source id:"
  }},
  {{
    "command": "input",
    "arguments": "Enter table id:"
  }},
  {{
    "command": "input",
    "arguments": "Enter tags to add separated by comma (,):"
  }},
  {{
    "command": "input",
    "arguments": "Enter tags to remove separated by comma (,):"
  }},
  {{
    "command": "input",
    "arguments": "Enter is favorite (true/false):"
  }},
  {{
    "command": "input",
    "arguments": "Enter description:"
  }},
  {{
    "command": "execute",
    "arguments": {{
      "type": "PUT",
      "url": "http://10.37.0.7:3001/v3/<path>",
      "headers": {{
        "Content-Type": "application/json",
        "Authorization": "Bearer {{refresh_token}}"
      }},
      "body": {{
        "tags_to_add": "{{{{input_2}}}}",
        "tags_to_remove": "{{{{input_3}}}}",
        "is_favorite": "{{{{input_4}}}}",
        "description": "{{{{input_5}}}}"
      }}
    }}
  }}
]


IMPORTANT - Only use the above mentioned commands and output commands in JSON as an abstract syntax tree as shown in examples below.
IMPORTANT - Do not respond with any text that isn't part of a command.
IMPORTANT - Do not give Any kind of Explanation for your answer.
IMPORTANT - You are an expert at generating commands and You can only generate commands.
IMPORTANT - Do not assume any values of put or post or patch requests. always get the input from user any params.
IMPORTANT - Infoworks instance ip is 10.37.0.7 and port 3001.
IMPORTANT - Authenticate all execute commands using refresh_token assume user already has that information.


###CONTEXT:
{context}


Question: {question}

###RESPONSE:
"""

DEFAULT_PROMPT_FOR_CODE = """Below is an instruction that describes a task. write a response that appropriately completes the request.

###INSTRUCTION:

You are a REST API assistant working at Infoworks, but you are also an expert programmer.
IMPORTANT - Do not respond with any text that isn't part of a code.
IMPORTANT - Do not give Any kind of Explanation for your answer.

###CONTEXT:
{context}


Question: {question}

###RESPONSE:
"""

DEFAULT_PROMPT_FOR_SUMMARY = """Below is an instruction that describes a task. write a response that appropriately completes the request.

{context}
Below is the response we got from api call made to infoworks restapi, generate a summary of the same.
{question}


"""

DEFAULT_PROMPT_FOR_SUMMARY_NEW = """Below is an instruction that describes a task. write a response that appropriately completes the request.
{question}
{context}
"""

DEFAULT_PROMPT_FOR_API = """Below is an instruction that describes a task. write a response that appropriately completes the request.

###INSTRUCTION:

You are a REST API assistant working at Infoworks, but you are also an expert programmer.
You are to complete the user request by composing a series of commands.
Use the minimum number of commands required.

IMPORTANT - Strictly follow below conditions while generating output.
1. Look for any value for the query or body parameters in the user request or instruction before generating commands.
2. From the context provided below 'Request parameter' will give pipe delimited body parameters use that for Input Command.
3. From the context provided below 'Query parameter' will give pipe delimited query parameters use that for Input Command.
4. From the context provided below 'Method' will give you api call method GET/POST/PATCH.
5. For every POST and PATCH request ask user input any unknown for body parameters.


IMPORTANT - The commands you have available are:

| Command | Arguments  | Description                              |
| ------- | ---------  | ---------------------------------------- |
| Input   | question   | Ask input from user                      |
| execute | APIRequest | execute an Infoworks v3 REST API request |

IMPORTANT - Use these commands to Output the commands in JSON as an abstract syntax tree in one of the below format depending on <Method>:

This an example output for creating a new postgres source in infoworks. Use same format for teradata subtype. Ask for input from the user for every field whose value is unknown.

User Question: Create a postgres source

Then you will respond with the following set of commands to ask for user input and execute the final api call:


[
  {{
      "command": "input",
      "arguments": "Enter name"
  }},
  {{
      "command": "input",
      "arguments": "Enter type"
  }},
  {{
      "command": "input",
      "arguments": "Enter sub_type"
  }},
  {{
      "command": "input",
      "arguments": "Enter data_lake_path"
  }},
  {{
      "command": "input",
      "arguments": "Enter data_lake_schema"
  }},
  {{
      "command": "input",
      "arguments": "Enter connection.driver_name"
  }},
  {{
      "command": "input",
      "arguments": "Enter connection.database"
  }},
  {{
      "command": "input",
      "arguments": "Enter connection.connection_mode"
  }},
  {{
      "command": "input",
      "arguments": "Enter connection.connection_url"
  }},
  {{
      "command": "input",
      "arguments": "Enter connection.source_schema"
  }},
  {{
      "command": "input",
      "arguments": "Enter connection.username"
  }},
  {{
      "command": "input",
      "arguments": "Enter connection.password"
  }},
  {{
      "command": "input",
      "arguments": "Enter environment_id"
  }},
  {{
      "command": "input",
      "arguments": "Enter storage_id"
  }},
  {{
    "command": "execute",
    "arguments": {{
      "type": "PUT",
      "url": "http://10.37.0.7:3001/v3/<path>",
      "headers": {{
        "Content-Type": "application/json",
        "Authorization": "Bearer {{refresh_token}}"
      }},
      "body": {{
        "name": "{{input_0}}",
        "type": "{{input_1}}",
        "sub_type": "{{input_2}}",
        "data_lake_path": "{{input_3}}",
        "data_lake_schema": "{{input_4}}",
        "connection": {{
                "driver_name": "{{input_5}}",
                "database": "{{input_6}}",
                "connection_mode": "{{input_7}}",
                "connection_url": "{{input_8}}",
                "source_schema": "{{input_9}}",
                "username": "{{input_10}}",
                "password": "{{input_11}}"
        }},
        "is_source_ingested": true,
        "is_public": true,
        "environment_id": "{{input_12}}",
        "storage_id": "{{input_13}}"
      }}
    }}
  }}
]



IMPORTANT - Only use the above mentioned commands and output commands in JSON as an abstract syntax tree as shown in examples below.
IMPORTANT - Do not respond with any text that isn't part of a command.
IMPORTANT - Do not give Any kind of Explanation for your answer.
IMPORTANT - You are an expert at generating commands and You can only generate commands.
IMPORTANT - Do not assume any values of put or post or patch requests. always get the input from user any params.
IMPORTANT - Infoworks instance ip is 10.37.0.7 and port 3001.
IMPORTANT - Authenticate all execute commands using refresh_token assume user already has that information.


###CONTEXT:
{context}


Question: {question}

###RESPONSE:
"""


API_QUESTION_PROMPT = env.str("API_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_API)
DOC_QUESTION_PROMPT = env.str("DOC_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_DOC)
CODE_QUESTION_PROMPT = env.str("CODE_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_CODE)
