"""
检查智能体生成的动作是否合规
"""
def check_action(action_plan):
    if action_plan[0].name == "none":
        if list(action_plan[1].keys())==[]:
            return True
        else:
            return False
    elif action_plan[0].name == "move_coords":
        if list(action_plan[1].keys()) == ['left','top']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['left']) == float and type(input_args_schema['top']) == float:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "click_coords":
        if list(action_plan[1].keys()) == ['left','top']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['left']) == float and type(input_args_schema['top']) == float:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "dbclick_coords":
        if list(action_plan[1].keys()) == ['left','top']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['left']) == float and type(input_args_schema['top']) == float:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "mousedown_coords":
        if list(action_plan[1].keys()) == ['left','top']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['left']) == float and type(input_args_schema['top']) == float:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "scroll_up_coords":
        if list(action_plan[1].keys()) == ['left','top','scroll_amount','scroll_time']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['left']) == float and type(input_args_schema['top']) == float:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "scroll_down_coords":
        if list(action_plan[1].keys()) == ['left','top','scroll_amount','scroll_time']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['left']) == float and type(input_args_schema['top']) == float:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "click_element":
        if list(action_plan[1].keys()) == ['ref']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['ref']) == int:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "press_key":
        if list(action_plan[1].keys()) == ['key']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['key']) == int:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "type_text":
        if list(action_plan[1].keys()) == ['text']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['text']) == str:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "type_field":
        if list(action_plan[1].keys()) == ['field']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['field']) == int:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "focus_element_and_type_text":
        if list(action_plan[1].keys()) == ['ref', 'text']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['ref']) == int and type(input_args_schema['text']) == str:
                return True
            else:
                return False
        else:
            return False
    elif action_plan[0].name == "focus_element_and_type_field":
        if list(action_plan[1].keys()) == ['ref', 'field']:
            input_args_schema = action_plan[1]
            if type(input_args_schema['ref']) == int and type(input_args_schema['field']) == int:
                return True
            else:
                return False
        else:
            return False
    else:
        return False