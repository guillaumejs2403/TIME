from random import sample
from functools import partial


# picture synonyms
picture = ["portrait", 'picture', 'image', 'photo', 'photograph', 'snapshot']


def glue_tokens(tokens):
    return ' '.join(tokens)


def generate_context_phrase(tokens):
    phrase = f'A {glue_tokens(tokens)} '
    phrase += sample(picture, 1)[0]
    return phrase


def generate_class_phrase(tokens,
                          base_prompt,
                          inbetween):
    phrase = f'{base_prompt} {inbetween} {glue_tokens(tokens)}'
    return phrase


def get_phrase_generator(args):
    '''
    Get phrase generator for the training
        :args: python object with the following attributes
               - args.base_prompt (str). Used for the class phase
    '''
    if args.dataset == 'CelebAHQ':
        if args.phase == 'context':
            return generate_context_phrase
        
        else:
            return partial(generate_class_phrase,
                           base_prompt=args.base_prompt,
                           inbetween='with a')

    elif args.dataset == 'BDD100k':
        if args.phase == 'context':
            return generate_context_phrase
        
        else:
            return partial(generate_class_phrase,
                           base_prompt=args.base_prompt,
                           inbetween='indicating to')

    else:
        raise ValueError(f'Dataset {args.dataset} not available')
