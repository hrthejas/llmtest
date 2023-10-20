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

DEFAULT_PROMPT_FOR_DOC = """

Use the below context and embeddings to answer the user questions

IMPORTANT - Infoworks is running on {base_url}.

CONTEXT: 
{context}
=========

QUESTION: {question} 

"""

DEFAULT_CSV_PARSE_ARGS = {
    "delimiter": ",",
    "fieldnames": ["Method", "Path", "Operation", "Description", "Query Parameters", "Request Parameters"],
}

CSV_DOC_PARSE_ARGS = env.dict("CSV_DOC_PARSE_ARGS", DEFAULT_CSV_PARSE_ARGS)

CSV_DOC_EMBEDDING_SOURCE_COLUMN = env.str("CSV_DOC_EMBEDDING_SOURCE_COLUMN", "Description")


DEFAULT_PROMPT_FOR_CODE = """Below is an instruction that describes a task. write a response that appropriately completes the request.

###INSTRUCTION:

You are a REST API assistant working at Infoworks, but you are also an expert programmer in python.
You are to complete the user request by writing code.

IMPORTANT - Always mak use of infoworks endpoints in the code for given task.
IMPORTANT - Response data is always will be in response.json()['result']
IMPORTANT - DO NOT assume any values for query_parameters either infer it from the previous response we go from previous call or take that as input from the user.
IMPORTANT - Always use '{base_url}/v3/' as base url for every infoworks endpoint.
IMPORTANT - Do not respond with any text that isn't part of a command.
IMPORTANT - Do not give Any kind of Explanation for your answer.
IMPORTANT - dont add Authenticate call,but include headers for every call, user already has bearer token with him and content type as json for the header
IMPORTANT - Try to parse response from previous api calls and see wht can be used as parameters to next api call if there are any. 
IMPORTANT - Only respond to requested ask do not add anything else, if the ask is to create a artifact just write code for that do not add code to list or edit etcc.

###CONTEXT:
{context}


Question: {question}

###RESPONSE:
"""

DEFAULT_PROMPT_FOR_SUMMARY = """Below is an instruction that describes a task. write a response that appropriately completes the request.

{context}
Below is the response we got from api call made to infoworks restapi, generate a summary of the same. Infoworks is running on {base_url}
{question}


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
6. Always start input index from 1 not 0


IMPORTANT - The commands you have available are:

| Command | Arguments  | Description                              |
| ------- | ---------  | ---------------------------------------- |
| Input   | question   | Ask input from user                      |
| execute | APIRequest | execute an Infoworks v3 REST API request |

IMPORTANT - Use these commands to Output the commands in JSON as an abstract syntax tree in the below format:


[
   //Ask this for every parameter
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
      "url": "{base_url}/v3/<path>",
      "headers": {{
        "Content-Type": "application/json",
        "Authorization": "Bearer {{refresh_token}}"
      }},
      "body": {{
        "name": "{{{{input_1}}}}",
        "type": "{{{{input_2}}}}",
        "sub_type": "{{{{input_3}}}}",
        "data_lake_path": "{{{{input_4}}}}",
        "data_lake_schema": "{{{{input_5}}}}",
        "connection": {{
                "driver_name": "{{{{input_6}}}}",
                "database": "{{{{input_7}}}}",
                "connection_mode": "{{{{input_8}}}}",
                "connection_url": "{{{{input_9}}}}",
                "source_schema": "{{{{input_10}}}}",
                "username": "{{{{input_11}}}}",
                "password": "{{{{input_12}}}}"
        }},
        "environment_id": "{{{{input_13}}}}",
        "storage_id": "{{{{input_14}}}}",
      }}
    }}
  }}
]



IMPORTANT - Only use the above mentioned commands and output commands in JSON as an abstract syntax tree as shown in examples below.
IMPORTANT - Do not respond with any text that isn't part of a command.
IMPORTANT - Do not give Any kind of Explanation for your answer.
IMPORTANT - You are an expert at generating commands and You can only generate commands.
IMPORTANT - Do not assume any values of put or post or patch requests. always get the input from user any params.
IMPORTANT - Infoworks is running on {base_url}.
IMPORTANT - Authenticate all execute commands using refresh_token assume user already has that information.


###CONTEXT:
{context}


Question: {question}

###RESPONSE:
"""

DEFAULT_PROMPT_FOR_API_HELP = """Below is an instruction that describes a task. write a response that appropriately completes the request.

###INSTRUCTION:

You are a REST API assistant working at Infoworks. Help the user by generating steps/docs to complete user request using infoworks endpoints

IMPORTANT - Response data is always will be in response.json()['result']
IMPORTANT - DO NOT assume any values for query_parameters either infer it from the previous response we go from previous call or take that as input from the user.
IMPORTANT - Always use '{base_url}/v3/' as base url for every infoworks endpoint.


###CONTEXT:
{context}


Question: {question}

###RESPONSE:
"""

DEFAULT_PROMPT_FOR_SQL_GEN = """
Below is an instruction that describes a task,paired with an input that provides further context. Write a response that appropriately completes the request.Do not add explanation or summary, only output SQL.

### INSTRUCTION:
You are an expert at writing sql consider following important points while generating sql
IMPORTANT - Do not add explanation or summary. Only output SQL
IMPORTANT - Refer to the schema information given below as create table statements for more information on the tables:
IMPORTANT - Refer to the each column description given as column level comments:

-- Table: CATEGORIES
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.CATEGORIES (
	CATEGORYID NUMBER(38,0),               -- ID for each product category.
	CATEGORYNAME VARCHAR(16777216),        -- Name of the product category.
	DESCRIPTION VARCHAR(16777216),         -- Brief description of the product category.
	PICTURE BINARY(8388608),               -- Image representing the product category.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when product category data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if product category is deleted or active
);

-- Table: CUSTOMERCUSTOMERDEMO
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.CUSTOMERCUSTOMERDEMO (
	CUSTOMERID VARCHAR(16777216),          -- ID for each customer.
	CUSTOMERTYPEID VARCHAR(16777216),      -- Type of customer.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when customer data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if customer data is deleted.
);

-- Table: CUSTOMERDEMOGRAPHICS
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.CUSTOMERDEMOGRAPHICS (
	CUSTOMERTYPEID VARCHAR(16777216),      -- Type of customer.
	CUSTOMERDESC VARCHAR(16777216),        -- Description of customer type.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when customer data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if customer data is deleted.
);

-- Table: CUSTOMERS
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.CUSTOMERS (
	CUSTOMERID VARCHAR(16777216),          -- ID for each customer.
	COMPANYNAME VARCHAR(16777216),         -- Company name of customer.
	CONTACTNAME VARCHAR(16777216),         -- Name of primary contact.
	CONTACTTITLE VARCHAR(16777216),        -- Role of primary contact.
	ADDRESS VARCHAR(16777216),             -- Customer's street address.
	CITY VARCHAR(16777216),                -- City where the customer is located.
	REGION VARCHAR(16777216),              -- Geographical region of customer. This could be the State in which the customer resides.
	POSTALCODE VARCHAR(16777216),          -- Postal code or ZIP code of customer's address.
	COUNTRY VARCHAR(16777216),             -- Country where customer operates.
	PHONE VARCHAR(16777216),               -- Contact phone number. This is their primary phone number.
	FAX VARCHAR(16777216),                 -- Contact fax number.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when customer data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if customer data is deleted.
);

-- Table: EMPLOYEES
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.EMPLOYEES (
	EMPLOYEEID NUMBER(38,0),               -- ID for each employee.
	LASTNAME VARCHAR(16777216),            -- Employee's last name.
	FIRSTNAME VARCHAR(16777216),           -- Employee's first name.
	TITLE VARCHAR(16777216),               -- Employee's job title.
	TITLEOFCOURTESY VARCHAR(16777216),     -- Polite form of address for the employee.
	BIRTHDATE TIMESTAMP_NTZ(9),            -- Employee's birthdate.
	HIREDATE TIMESTAMP_NTZ(9),             -- Date when employee was hired.
	ADDRESS VARCHAR(16777216),             -- Employee's street address.
	CITY VARCHAR(16777216),                -- City where employee lives.
	REGION VARCHAR(16777216),              -- Larger region where employee resides. This could be the State where Employee resides.
	POSTALCODE VARCHAR(16777216),          -- Postal code or ZIP code of employee's address.
	COUNTRY VARCHAR(16777216),             -- Country where employee lives.
	HOMEPHONE VARCHAR(16777216),           -- Employee's home phone number.
	EXTENSION VARCHAR(16777216),           -- Phone extension for employee.
	PHOTO BINARY(8388608),                 -- Photo of the employee.
	NOTES VARCHAR(16777216),               -- Additional notes about employee.
	REPORTSTO NUMBER(38,0),                -- ID of supervisor employee.
	PHOTOPATH VARCHAR(16777216),           -- Path to employee's photo.
	SALARY FLOAT,                          -- Employee's salary.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when employee data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if employee data is deleted.
);

-- Table: EMPLOYEETERRITORIES
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.EMPLOYEETERRITORIES (
	EMPLOYEEID NUMBER(38,0),               -- ID of employee.
	TERRITORYID VARCHAR(16777216),         -- ID of territory assigned to employee.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if data is deleted.
);

-- Table: ORDERS
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.ORDERS (
	ORDERID NUMBER(38,0),                   -- Unique identifier for each order.
	CUSTOMERID VARCHAR(16777216),           -- Identifier of the customer.
	EMPLOYEEID NUMBER(38,0),                -- Identifier of the employee who took the order.
	ORDERDATE TIMESTAMP_NTZ(9),             -- Date when the order was placed.
	REQUIREDDATE TIMESTAMP_NTZ(9),          -- Date when the order is needed.
	SHIPPEDDATE TIMESTAMP_NTZ(9),           -- Date when the order was shipped.
	SHIPVIA NUMBER(38,0),                   -- Identifier of the shipping company.
	FREIGHT NUMBER(10,4),                   -- Freight Cost of shipping.
	SHIPNAME VARCHAR(16777216),             -- Name of the recipient.
	SHIPADDRESS VARCHAR(16777216),          -- Shipping address.
	SHIPCITY VARCHAR(16777216),             -- Shipping city.
	SHIPREGION VARCHAR(16777216),           -- Shipping region. This could be the State for Shipping.
	SHIPPOSTALCODE VARCHAR(16777216),       -- Shipping postal code or Shipping ZIP code.
	SHIPCOUNTRY VARCHAR(16777216),          -- Shipping country.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if data is deleted.
);

-- Table: ORDER_DETAILS
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.ORDER_DETAILS (
	ORDERID NUMBER(38,0),                   -- Identifier for the order.
	PRODUCTID NUMBER(38,0),                 -- Identifier for the product in the order.
	UNITPRICE NUMBER(10,4),                 -- Price per unit of the product. This is in $.
	QUANTITY NUMBER(38,0),                  -- Number of units ordered.
	DISCOUNT FLOAT,                         -- Discount applied to the entire order's $ value. This is specified as a percentage value. Total order value is calculated as Quantity * Unit Price and then reducing the discount from that value.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if data is deleted.
);

-- Table: PRODUCTS
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.PRODUCTS (
	PRODUCTID NUMBER(38,0),                 -- Identifier for each product.
	PRODUCTNAME VARCHAR(16777216),          -- Name of the product.
	SUPPLIERID NUMBER(38,0),                -- Identifier of the supplier.
	CATEGORYID NUMBER(38,0),                -- Identifier of the category the product belongs to.
	QUANTITYPERUNIT VARCHAR(16777216),      -- Quantity and measurement description per unit.
	UNITPRICE NUMBER(10,4),                 -- Price per unit of the product.
	UNITSINSTOCK NUMBER(38,0),              -- Number of units currently in stock.
	UNITSONORDER NUMBER(38,0),              -- Number of units on order.
	REORDERLEVEL NUMBER(38,0),              -- Minimum number of units to trigger a reorder.
	DISCONTINUED BOOLEAN,                   -- Indicates if the product is no longer available in stock or not being supplied by the vendor or has been marked as inactive.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if product data is deleted.
);

-- Table: REGION
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.REGION (
	REGIONID NUMBER(38,0),                  -- Identifier for each geographical region.
	REGIONDESCRIPTION VARCHAR(16777216),    -- Description of the region.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if data is deleted.
);

-- Table: SHIPPERS
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.SHIPPERS (
	SHIPPERID NUMBER(38,0),                 -- Identifier for each shipping company.
	COMPANYNAME VARCHAR(16777216),          -- Name of the shipping company.
	PHONE VARCHAR(16777216),                -- Contact phone number.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if data is deleted.
);

-- Table: SUPPLIERS
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.SUPPLIERS (
	SUPPLIERID NUMBER(38,0),                -- Identifier for each supplier.
	COMPANYNAME VARCHAR(16777216),          -- Name of the supplier company.
	CONTACTNAME VARCHAR(16777216),          -- Name of the contact person.
	CONTACTTITLE VARCHAR(16777216),         -- Job title of the contact person.
	ADDRESS VARCHAR(16777216),              -- Supplier's street address.
	CITY VARCHAR(16777216),                 -- City where supplier is located.
	REGION VARCHAR(16777216),               -- Larger region where supplier is situated. This could be State where the supplier is located.
	POSTALCODE VARCHAR(16777216),           -- Postal code or ZIP code of supplier's address.
	COUNTRY VARCHAR(16777216),              -- Country where supplier operates.
	PHONE VARCHAR(16777216),                -- Contact phone number.
	FAX VARCHAR(16777216),                  -- Contact fax number.
	HOMEPAGE VARCHAR(16777216),             -- Supplier's homepage or website.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if supplier data is deleted.
);

-- Table: TERRITORIES
create or replace TABLE NORTHWIND_AI_DB.NORTHWIND_AI_SCHEMA.TERRITORIES (
	TERRITORYID VARCHAR(16777216),          -- Identifier for each territory.
	TERRITORYDESCRIPTION VARCHAR(16777216), -- Description of the territory.
	REGIONID NUMBER(38,0),                  -- Identifier of the region the territory is in.
	ZIW_TARGET_TIMESTAMP TIMESTAMP_NTZ(9), -- Time when data was added.
	ZIW_IS_DELETED BOOLEAN                 -- Indicates if data is deleted.
);

Follow the instructions above and Convert the following text to snowflake compatible sql: {user_text}

###RESPONSE:
"""

API_QUESTION_PROMPT = env.str("API_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_API)
DOC_QUESTION_PROMPT = env.str("DOC_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_DOC)
CODE_QUESTION_PROMPT = env.str("CODE_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_CODE)
SUMMARY_QUESTION_PROMPT = env.str("SUMMARY_QUESTION_PROMPT",DEFAULT_PROMPT_FOR_SUMMARY)
API_HELP_QUESTION_PROMPT = env.str("API_HELP_QUESTION_PROMPT", DEFAULT_PROMPT_FOR_API_HELP)

DEFAULT_PROMPT_FOR_SQL_GEN_BKP = """
Below is an instruction that describes a task,paired with an input that provides further context. Write a response that appropriately completes the request.Do not add explanation or summary, only output SQL.

### INSTRUCTION:
You are an expert at writing sql consider following important points while generating sql
IMPORTANT - Do not add explanation or summary. Only output SQL
IMPORTANT - Refer to the schema information given below more information on the tables:
IMPORTANT - Refer to the each column description given as column level comments:


categories : categoryid,categoryname,description,picture 
customercustomerdemo : customerid,customertypeid
customerdemographics : customertypeid,customerdesc 
customers : customerid,companyname,contactname,contacttitle,address,city,region,postalcode,country,phone,fax 
employees : employeeid,lastname,firstname,title,titleofcourtesy,birthdate,hiredate,address,city,region,postalcode,country,homephone,extension,photo,notes,reportsto,photopath,salary
employeeterritories : employeeid,territoryid
order_details : orderid,productid,unitprice,quantity,discount
orders : orderid,customerid,employeeid,orderdate,requireddate,shippeddate,shipvia,freight,shipname,shipaddress,shipcity,shipregion,shippostalcode,shipcountry
products : productid,productname,supplierid,categoryid,quantityperunit,unitprice,unitsinstock,unitsonorder,reorderlevel,discontinued
region : regionid,regiondescription
shippers : shipperid,companyname,phone
suppliers : supplierid,companyname,contactname,contacttitle,address,city,region,postalcode,country,phone,fax,homepage
territories : territoryid,territorydescription,regionid
customercustomerdemo.customertypeid = customerdemographics.customertypeid
customercustomerdemo.customerid = customers.customerid
orders.customerid = customers.customerid
orders.shipvia = shippers.shipperid
orders.employeeid = employees.employeeid
employeeterritories.employeeid = employees.employeeid
employeeterritories.territoryid = territories.territoryid
territories.regionid = region.regionid
order_details.orderid = orders.orderid
order_details.productid = products.productid
products.supplierid = suppliers.supplierid
products.categoryid = categories.categoryid


Follow the instructions above and Convert the following text to sql: {user_text}

###RESPONSE:
"""

DEFAULT_PROMPT_FOR_DASHBOARD = """
Below is an instruction that describes a task,paired with an input that provides further context. Write a response that appropriately completes the request.Do not add explanation or summary, only output SnowSQL with its heading.

### INSTRUCTION:
You are business analyst and you are very good at Snowflake SQL. you need to generate sql for building dashboard.
Please output only SnowSQL for sales performance dashboard  using the below provided SQL statement. Generate SQL queries for generate dashboard that highlights the top 5 products by sales, customer retention rate, and sales growth rate to measure customer loyalty and business growth. With detailed insights on sales by sales channel, month, region, online orders, store orders, products sold, and sales by customer segment, this dashboard empowers businesses to make data-driven decisions and optimize their sales strategies.

use the following sql statement to generate dashboard snowflake SQL statement as run against.
SQL: 
CREATE OR REPLACE TABLE SalesPerformanceWide AS
SELECT
    p.ProductName,
    o.OrderDate,
    o.Region,
    o.SalesChannel,
    o.CustomerSegment,
    SUM(od.UnitsOrdered) AS TotalUnitsOrdered,
    SUM(od.UnitsSold) AS TotalUnitsSold,
    SUM(od.Revenue) AS TotalRevenue,
    AVG(od.UnitsPerTransaction) AS AvgUnitsPerTransaction,
    AVG(od.OrderValue) AS AvgOrderValue,
    AVG(od.SalesCycleTime) AS AvgSalesCycleTime,
    AVG(od.ConversionRate) AS AvgConversionRate,
    AVG(od.ProductToSalesRatio) AS AvgProductToSalesRatio
FROM
    Products p
JOIN
    OrderDetails od
ON
    p.ProductID = od.ProductID
JOIN
    Orders o
ON
    od.OrderID = o.OrderID
GROUP BY
    p.ProductName,
    o.OrderDate,
    o.Region,
    o.SalesChannel,
    o.CustomerSegment;

Note: Only print SnowSQL with its heading and no other text at all.
Note: Genetate SQL for aggregations or filters as needed for the dashboard.
Note: Your SQL statements will use the "SalesPerformanceWide" table to generate the desired reports.

###RESPONSE:
"""