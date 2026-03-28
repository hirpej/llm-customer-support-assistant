# LLM-Based Customer Support Assistant with Tool Calling

An intelligent assistant that uses LLM function calling to automate customer support tasks such as order lookup and refund processing.

## Overview

This project demonstrates how a large language model can interact with structured tools to execute real-world operations.

The assistant analyzes user queries, decides which tool to call, executes the tool, and returns a structured and human-readable response.

## Features

- LLM-based tool selection (function calling)
- Structured tool execution (e.g., order lookup, refunds)
- Multi-step reasoning workflow
- JSON-based responses
- Tool call logging for debugging and transparency
- Interactive Streamlit interface

## How It Works

1. User submits a query  
2. LLM determines whether a tool is needed  
3. If required, the model generates a tool call  
4. The system executes the tool with parsed arguments  
5. Tool results are returned to the model  
6. The model generates a final response  

## Project Structure
