import os
import sys
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy

# 0. Functions to load the model

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # Here is the attention function
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def length_constrained_decode(model, src, src_mask, src_length, start_symbol, end_symbol, pad_symbol, vocab_tgt):

    memory = model.encode(src, src_mask)
    # Inicializar con el token de inicio
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    
    # Crear un diccionario inverso para acceder a los tokens por ID
    vocab_dict = {i: token for i, token in enumerate(vocab_tgt.get_itos())}
    
    # Tokens especiales
    special_tokens = [start_symbol, end_symbol, pad_symbol]
    
    # Inicializar contador de caracteres generados
    generated_length = 0
    
    # Almacenar secuencia de tokens generados (para debugging)
    generated_tokens = []
    
    # Establecer un límite de iteraciones para evitar bucles infinitos
    max_iterations = 1000
    iterations = 0
    
    while generated_length < src_length and iterations < max_iterations:
        iterations += 1
        
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        
        # Obtener la distribución de probabilidad para el siguiente token
        prob = model.generator(out[:, -1])
        
        # Si estamos a un solo carácter de completar la longitud requerida
        if src_length - generated_length == 1:
            # Como no hay tokens de un carácter, terminamos aquí y ajustamos después
            print("Only one character is missing. Decoding is ending and will be adjusted later..")
            ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(end_symbol)], dim=1)
            generated_tokens.append((end_symbol, vocab_dict.get(end_symbol, ""), 0))
            break
        else:
            # Seleccionar el token más probable
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            print(f"Generated token: {next_word} ({vocab_tgt.get_itos()[next_word]})")
            print(f"Generated length: {generated_length}, Target length: {src_length}")
        
        # Obtener la representación del token y su longitud
        token_str = vocab_dict.get(next_word, "")
        # Calcular la longitud real (sin tokens especiales)
        token_length = len(eliminar_tokens_especiales(token_str))
        
        # Verificar si al agregar este token superaríamos la longitud deseada
        if generated_length + token_length > src_length:
            # Buscar un token que complete exactamente la longitud
            remaining_length = src_length - generated_length
            suitable_token = None
            
            # Buscar entre los tokens más probables uno que tenga la longitud exacta requerida
            top_k_values, top_k_indices = torch.topk(prob[0], k=min(100, prob.size(1)))
            for idx in top_k_indices:
                idx_token = idx.item()
                if idx_token in special_tokens:
                    continue
                token_text = vocab_dict.get(idx_token, "")
                clean_length = len(eliminar_tokens_especiales(token_text))
                if clean_length == remaining_length:
                    suitable_token = idx_token
                    break
            
            # Si encontramos un token adecuado, usarlo
            if suitable_token is not None:
                next_word = suitable_token
                print(f"Generated token: {next_word} ({vocab_tgt.get_itos()[next_word]})")
                print(f"Generated length: {generated_length}, Target length: {src_length}")
            else:
                # En última instancia, forzar el token de fin
                next_word = end_symbol
                
        # Agregar el token seleccionado a la secuencia
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        generated_tokens.append((next_word, vocab_dict.get(next_word, ""), token_length))
        
        # Actualizar la longitud generada si no es un token especial
        if next_word not in special_tokens:
            generated_length += token_length
            
        # Si ya alcanzamos la longitud exacta y el último token no es el de fin, agregar token de fin
        if generated_length == src_length and next_word != end_symbol:
            ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(end_symbol)], dim=1)
            generated_tokens.append((end_symbol, vocab_dict.get(end_symbol, ""), 0))
            break
    
    # Verificar si se alcanzó el límite de iteraciones
    if iterations >= max_iterations:
        print("¡Advertencia! Se alcanzó el número máximo de iteraciones.")
        # Forzar finalización con token de fin
        ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(end_symbol)], dim=1)
    
    return ys


################

# 1. Functions to be able to do all the processing

import re

# Define the general mapping dictionary
mapeo_original = {
    'A(': 'Q',
    'C(': 'W',
    'G(': 'R',
    'U(': 'Y',
    'A.': 'I',
    'C.': 'V',
    'G.': 'P',
    'U.': 'S',
    'A)': 'H',
    'C)': 'J',
    'G)': 'K',
    'U)': 'L',
    'A<': '1',
    'C<': '2',
    'G<': '3',
    'U<': '4',
    'A>': '5',
    'C>': '6',
    'G>': '7',
    'U>': '8',
    'A{': 'D',
    'C{': 'F',
    'G{': 'Z',
    'U{': 'X',
    'A}': 'B',
    'C}': 'M',
    'G}': 'O',
    'U}': '0',
}

mapeo = {**mapeo_original, **{k[0].lower() + k[1]: v for k, v in mapeo_original.items()}}

# Function to concatenate 1D sequence and 2D structure
def concatenar_entradas(seq, dot_bracket_result):
    if len(seq) == 0 or len(dot_bracket_result) == 0 or len(seq) != len(dot_bracket_result):
        print("Error: Las secuencias son de diferente longitud o están vacías.")
        return None

    resultado = []
    for char_seq, char_struct in zip(seq, dot_bracket_result):
        concatenacion = char_seq + char_struct
        resultado.append(concatenacion)

    return resultado

# Function to encode the result of concatenation
def codificar_resultado(resultado):
    if len(resultado) == 0:
        print("Error: El resultado está vacío.")
        return None

    nueva_secuencia = ""
    for elem in resultado:
        if elem in mapeo:
            nueva_secuencia += mapeo[elem]
        else:
            nueva_secuencia += elem

    return nueva_secuencia

def eliminar_espacios_y_tokens(texto):
    return texto.replace(" ", "").replace("<s>", "").replace("</s>", "")

mapeo_deco = {
    'A(': 'Q', 'C(': 'W', 'G(': 'R', 'U(': 'Y',
    'A.': 'I', 'C.': 'V', 'G.': 'P', 'U.': 'S',
    'A)': 'H', 'C)': 'J', 'G)': 'K', 'U)': 'L',
    'A<': '1', 'C<': '2', 'G<': '3', 'U<': '4',
    'A>': '5', 'C>': '6', 'G>': '7', 'U>': '8',
    'A{': 'D', 'C{': 'F', 'G{': 'Z', 'U{': 'X',
    'A}': 'B', 'C}': 'M', 'G}': 'O', 'U}': '0',
}

def decodificar_resultado(secuencia_codificada):
    decodificado = ""
    for char in secuencia_codificada:
        for key, value in mapeo_deco.items():
            if value == char:
                decodificado += key[-1]
                break
        else:
            decodificado += char
    return decodificado


# Function to remove special tokens and spaces
def eliminar_tokens_especiales(texto):
    """Removes special tokens and spaces from the text."""
    return texto.replace(" ", "").replace("<s>", "").replace("</s>", "").replace("<blank>", "")


# Define the two kinds of tokenization

class SimpleSpacy_T1:
    def __init__(self):
        pass

    def tokenize(self, text):
        text = text.strip()  # Remove leading/trailing whitespace including newlines
        tokens = []
        start = 0
        groups = [('QWRY', 2), ('IVPS', 2), ('HJKL', 2), ('1234', 2), ('5678', 2), ('DFZX', 2), ('BMO0', 2)]

        while start < len(text):
            found_group = False
            for group, min_length in groups:
                if start < len(text) and text[start] in group:
                    end = start + 1
                    while end < len(text) and text[end] in group:
                        end += 1

                    if end - start >= min_length:
                        new_token = text[start:end]
                        if not tokens:
                            tokens.append(new_token)
                        elif tokens[-1] in group:
                            tokens[-1] += new_token
                        else:
                            tokens.append(new_token)

                        start = end
                        found_group = True
                        break

            if not found_group:
                if len(tokens) > 0 and tokens[-1] != ' ':
                    tokens[-1] += text[start]
                elif text[start] != ' ':
                    tokens.append(text[start])
                start += 1

        i = 0
        while i < len(tokens) - 1:
            if len(tokens[i]) == 1 and tokens[i] != ' ':
                tokens[i+1] = tokens[i] + tokens[i+1]
                tokens.pop(i)
            else:
                i += 1

        return tokens



def count_pseudoKnots(sequence):
    knots = []
    current_knot_size = 0
    in_knot = False

    for char in sequence:
        if char == "<":
            current_knot_size += 1
            in_knot = True
        else:
            if in_knot:
                if char in [".", "("]:
                    continue
                else:
                    knots.append(current_knot_size)
                    current_knot_size = 0
                    in_knot = False

    if current_knot_size > 0:
        knots.append(current_knot_size)

    return knots

def chequeo_match_de_parentesis(sequence):
    def check_and_correct(seq, open_char, close_char):
        stack = []
        to_replace = []
        for i, char in enumerate(seq):
            if char == open_char:
                stack.append(i)
            elif char == close_char:
                if stack:
                    stack.pop()
                else:
                    to_replace.append(i)

        to_replace.extend(stack)
        return to_replace

    parentheses_to_replace = check_and_correct(sequence, '(', ')')
    angles_to_replace = check_and_correct(sequence, '<', '>')

    if not parentheses_to_replace and not angles_to_replace:
        print("\nAll types of parentheses are well matched")
        return sequence

    print("\nThere are parentheses that are not well matched.. starting arrangement of parentheses that are not well matched..")

    result = list(sequence)

    for index in parentheses_to_replace:
        print(f"Reemplazando '{result[index]}' en la posición {index} por '.''")
        result[index] = '.'

    for index in angles_to_replace:
        print(f"Reemplazando '{result[index]}' en la posición {index} por '.''")
        result[index] = '.'

    return ''.join(result)


#######################################################

# 2. Function to run the model

def run_model_example(input_str, vocab_src, vocab_tgt, model_path, device='cpu'):
    """Run the model for prediction"""
    # Initialize tokenizer
    tokenizer = SimpleSpacy_T1()

    # Tokenize the input
    tokenized_input = tokenizer.tokenize(input_str)
    print("Model Input: ", tokenized_input)
    
    # Original input length without spaces
    input_length = len(input_str.replace(" ", ""))
    print(f"Original input length (without spaces): {input_length}")

    # Convert tokens to ids
    src_tokens = [vocab_src.get_stoi().get(token, vocab_src.get_stoi()['<unk>']) for token in tokenized_input]

    # Add special tokens
    start_symbol = vocab_tgt.get_stoi()['<s>']
    end_symbol = vocab_tgt.get_stoi()['</s>']
    pad_symbol = vocab_tgt.get_stoi()['<blank>']
    src_tokens = [start_symbol] + src_tokens + [end_symbol]

    # Convert to tensor
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

    # Create mask
    src_mask = (src_tensor != vocab_src.get_stoi()['<blank>']).unsqueeze(-2)

    print("Loading Trained Model ...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded in {model.training} mode")

    # Generate prediction
    
    with torch.no_grad():
       model_output = length_constrained_decode(
           model, 
           src_tensor, 
           src_mask, 
           input_length,  # Longitud exacta que debe tener en caracteres
           start_symbol,
           end_symbol,
           pad_symbol,
           vocab_tgt  # Pasar el vocabulario completo
       )

    # Convert prediction to text
    model_text = ' '.join([vocab_tgt.get_itos()[token] for token in model_output[0].tolist() if token != pad_symbol])

    print("Model Output: ", model_text)
    
    # Remove spaces and special tokens to verify length
    texto_limpio = eliminar_tokens_especiales(model_text)
    print(f"Output length: {len(texto_limpio)}")
    
    # Verify lengths match
    if len(texto_limpio) != input_length:
        print(f"WARNING! Lengths do not match: Input={input_length}, Output={len(texto_limpio)}")
        
        # Adjust length if exactly one character is missing
        if len(texto_limpio) == input_length - 1:
            print("Adding the character 'I' to complete the exact length")
            texto_limpio += 'I'
            print(f"New output length: {len(texto_limpio)}")
            # Recreate model_text with added character
            model_text += ' I'
            print("Adjusted Model Output: ", model_text)
    
    print("Lengths match correctly." if len(texto_limpio) == input_length else "Lengths still do not match.")
    
    return model_text


##########################################

def find_base_pairs(structure):
    """
    Encuentra todos los pares de bases en una estructura RNA.
    
    Args:
        structure (str): Estructura RNA en notación de paréntesis
    
    Returns:
        dict: Diccionario con pares regulares '()' y pseudoknots '<>'
    """
    regular_pairs = []  # Pares '()'
    pseudoknot_pairs = []  # Pares '<>'
    
    # Encontrar pares '()'
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                regular_pairs.append((stack.pop(), i))
    
    # Encontrar pares '<>'
    stack = []
    for i, char in enumerate(structure):
        if char == '<':
            stack.append(i)
        elif char == '>':
            if stack:
                pseudoknot_pairs.append((stack.pop(), i))
    
    return {'regular': regular_pairs, 'pseudoknot': pseudoknot_pairs}

####aca

def combine_structures_direct(probknot_structure, clarafold_structure):
    """
    Método directo para combinar estructuras de RNA preservando los pseudonudos de ClaraFold
    mientras mantiene la mayor cantidad posible de pares regulares de ProbKnot.
    Preserva bloques completos de pseudonudos incluyendo puntos intermedios.
    Si ClaraFold no contiene pseudonudos, elimina todos los pseudonudos de ProbKnot.
    Si ambas estructuras tienen la misma cantidad de pseudonudos, conserva ProbKnot tal como está.
    
    Args:
        probknot_structure (str): Estructura de ProbKnot
        clarafold_structure (str): Estructura de ClaraFold con pseudonudos
        
    Returns:
        str: Estructura combinada con pseudonudos preservados o eliminados según corresponda
    """
    # Verificar longitudes
    if len(probknot_structure) != len(clarafold_structure):
        raise ValueError("Las estructuras deben tener la misma longitud")
    
    # Contar pseudonudos en ambas estructuras
    probknot_count, _ = count_pseudoknots(probknot_structure)
    clarafold_count, _ = count_pseudoknots(clarafold_structure)
    
    # Si ambas estructuras tienen la misma cantidad de pseudonudos, conservar ProbKnot
    if probknot_count == clarafold_count:
        return probknot_structure
    
    # Verificar si ClaraFold tiene pseudonudos
    has_pseudoknots_clarafold = '<' in clarafold_structure or '>' in clarafold_structure
    
    # Eliminar siempre todos los pseudonudos de ProbKnot (representados con '<>', '{}')
    probknot_no_pseudoknots = ''.join(['.' if char in '<>{}' else char for char in probknot_structure])
    
    # Si ClaraFold no tiene pseudonudos, retornar ProbKnot sin pseudonudos
    if not has_pseudoknots_clarafold:
        return probknot_no_pseudoknots
    
    # Identificar bloques de pseudonudos en ClaraFold
    pseudoknot_blocks = identify_pseudoknot_blocks(clarafold_structure)
    
    # Inicializar con la estructura de ProbKnot sin pseudonudos
    combined = list(probknot_no_pseudoknots)
    
    # Marcar todas las posiciones que forman parte de bloques de pseudonudos
    pseudoknot_positions = set()
    for start, end in pseudoknot_blocks:
        for pos in range(start, end + 1):
            pseudoknot_positions.add(pos)
    
    # Insertar los bloques completos de pseudonudos de ClaraFold
    for start, end in pseudoknot_blocks:
        for pos in range(start, end + 1):
            combined[pos] = clarafold_structure[pos]
    
    # Identificar pares de bases regulares en ProbKnot
    regular_pairs = []
    regular_stack = []
    
    for i, char in enumerate(probknot_structure):
        if char == '(':
            regular_stack.append(i)
        elif char == ')':
            if regular_stack:
                open_pos = regular_stack.pop()
                regular_pairs.append((open_pos, i))
    
    # Revisar y reparar inconsistencias causadas por pseudonudos
    for open_pos, close_pos in regular_pairs:
        # Si un par regular tiene una posición ya ocupada por un pseudonudo o punto dentro de un bloque,
        # necesitamos eliminar ambos extremos del par regular
        if open_pos in pseudoknot_positions or close_pos in pseudoknot_positions:
            # Solo eliminar si no fueron ya reemplazados por el bloque de pseudonudos
            if combined[open_pos] == '(':
                combined[open_pos] = '.'
            if combined[close_pos] == ')':
                combined[close_pos] = '.'
    
    # Verificar si hay pseudonudos sin emparejar después de la combinación
    return fix_pairing_issues(''.join(combined))

def identify_pseudoknot_blocks(structure):
    """
    Identifica bloques continuos que contienen pseudonudos, incluyendo puntos intermedios.
    Un bloque comienza con el primer '<' o '>' y termina cuando hay una secuencia
    de al menos 2 caracteres que no son '<', '>' ni puntos.
    
    Args:
        structure (str): Estructura RNA
        
    Returns:
        list: Lista de tuplas (inicio, fin) de cada bloque de pseudonudos
    """
    blocks = []
    i = 0
    length = len(structure)
    
    while i < length:
        # Buscar el inicio de un bloque (primer '<' o '>')
        if structure[i] in '<>':
            start = i
            # Avanzar mientras encontremos '<', '>' o puntos
            # con no más de 1 punto consecutivo entre símbolos de pseudonudos
            consecutive_dots = 0
            last_pseudoknot_pos = i
            
            while i < length:
                if structure[i] in '<>':
                    consecutive_dots = 0
                    last_pseudoknot_pos = i
                elif structure[i] == '.':
                    consecutive_dots += 1
                    # Si hay más de 1 punto consecutivo y estamos lejos del último pseudonudo,
                    # consideramos que es el fin del bloque
                    if consecutive_dots > 1 and (i - last_pseudoknot_pos) > 1:
                        break
                else:
                    # Si encontramos otro carácter, es el fin del bloque
                    break
                i += 1
            
            # El fin del bloque es la última posición de pseudonudo encontrada
            end = last_pseudoknot_pos
            
            # Solo guardar bloques que tengan al menos un pseudonudo
            if start <= end:
                blocks.append((start, end))
        else:
            i += 1
            
    return blocks

def fix_pairing_issues(structure):
    """
    Corrige problemas de emparejamiento en la estructura.
    Asegura que todos los paréntesis y pseudonudos estén correctamente emparejados.
    
    Args:
        structure (str): Estructura a verificar y corregir
        
    Returns:
        str: Estructura corregida
    """
    # Manejar pares regulares
    result = list(structure)
    
    # Verificar pares regulares
    regular_stack = []
    for i, char in enumerate(structure):
        if char == '(':
            regular_stack.append(i)
        elif char == ')':
            if regular_stack:
                regular_stack.pop()
            else:
                result[i] = '.'  # Cerrar sin abrir
    
    for pos in regular_stack:
        result[pos] = '.'  # Abrir sin cerrar
    
    # Verificar pseudonudos
    pk_stack = []
    for i, char in enumerate(''.join(result)):
        if char == '<':
            pk_stack.append(i)
        elif char == '>':
            if pk_stack:
                pk_stack.pop()
            else:
                result[i] = '.'  # Cerrar sin abrir
    
    for pos in pk_stack:
        result[pos] = '.'  # Abrir sin cerrar
    
    return ''.join(result)

def extract_pseudoknot_groups(structure):
    """
    Extrae grupos consecutivos de pseudonudos de una estructura.
    
    Args:
        structure (str): Estructura de RNA
        
    Returns:
        list: Lista de tuplas (inicio, fin, contenido) de cada grupo
    """
    groups = []
    in_group = False
    start = 0
    group_content = ""
    
    for i, char in enumerate(structure):
        if char in '<>' and not in_group:
            in_group = True
            start = i
            group_content = char
        elif char in '<>' and in_group:
            group_content += char
        elif char not in '<>' and in_group:
            if len(group_content) > 0:  # Solo guardar si hay contenido real
                groups.append((start, i-1, group_content))
            in_group = False
            group_content = ""
    
    # Añadir el último grupo si estamos dentro de uno
    if in_group and len(group_content) > 0:
        groups.append((start, len(structure)-1, group_content))
    
    return groups

def count_pseudoknots(structure):
    """
    Cuenta el número de pseudonudos en una estructura de ARN, manejando tanto símbolos '<>' como '{}'
    y permitiendo puntos y otros caracteres entre los símbolos de apertura y cierre.
    Esta versión maneja correctamente pseudonudos entrelazados.
    
    Args:
        structure (str): String de estructura de ARN
        
    Returns:
        tuple: (número de pseudonudos, lista de pseudonudos encontrados)
    """
    # Identificar bloques de pseudonudos
    angle_blocks = identify_pseudoknot_sequences(structure, '<', '>')
    curly_blocks = identify_pseudoknot_sequences(structure, '{', '}')
    
    # Lista para almacenar pseudonudos completos identificados
    pseudoknots = []
    pseudoknot_count = 0
    
    # Procesar bloques de tipo ángulo '<>'
    matched_pseudoknots = match_pseudoknot_blocks(structure, angle_blocks['open'], angle_blocks['close'])
    pseudoknots.extend(matched_pseudoknots)
    pseudoknot_count += len(matched_pseudoknots)
    
    # Procesar bloques de tipo llave '{}'
    matched_pseudoknots = match_pseudoknot_blocks(structure, curly_blocks['open'], curly_blocks['close'])
    pseudoknots.extend(matched_pseudoknots)
    pseudoknot_count += len(matched_pseudoknots)
    
    return pseudoknot_count, pseudoknots

def identify_pseudoknot_sequences(structure, open_char, close_char):
    """
    Identifica secuencias de símbolos de apertura y cierre, incluyendo puntos intermedios.
    
    Args:
        structure (str): String de estructura de ARN
        open_char (str): Carácter de apertura ('<' o '{')
        close_char (str): Carácter de cierre ('>' o '}')
        
    Returns:
        dict: Diccionario con listas de bloques de apertura y cierre
    """
    open_blocks = []  # Lista para almacenar bloques de apertura
    close_blocks = []  # Lista para almacenar bloques de cierre
    
    i = 0
    while i < len(structure):
        # Buscar secuencias de apertura
        if structure[i] == open_char:
            start = i
            count = 0  # Contar símbolos de apertura en este bloque
            
            while i < len(structure) and (structure[i] == open_char or structure[i] == '.'):
                if structure[i] == open_char:
                    count += 1
                i += 1
            
            # Solo guardamos bloques que tengan al menos un símbolo de apertura
            if count > 0:
                open_blocks.append({
                    'start': start,
                    'end': i - 1,
                    'count': count,
                    'sequence': structure[start:i]
                })
                continue  # Continuamos desde la posición actual
        
        # Buscar secuencias de cierre
        elif structure[i] == close_char:
            start = i
            count = 0  # Contar símbolos de cierre en este bloque
            
            while i < len(structure) and (structure[i] == close_char or structure[i] == '.'):
                if structure[i] == close_char:
                    count += 1
                i += 1
            
            # Solo guardamos bloques que tengan al menos un símbolo de cierre
            if count > 0:
                close_blocks.append({
                    'start': start,
                    'end': i - 1,
                    'count': count,
                    'sequence': structure[start:i]
                })
                continue  # Continuamos desde la posición actual
        
        i += 1
    
    return {'open': open_blocks, 'close': close_blocks}

def match_pseudoknot_blocks(structure, open_blocks, close_blocks):
    """
    Empareja bloques de apertura y cierre para identificar pseudonudos completos.
    
    Args:
        structure (str): String de estructura de ARN
        open_blocks (list): Lista de bloques de apertura
        close_blocks (list): Lista de bloques de cierre
        
    Returns:
        list: Lista de pseudonudos identificados
    """
    matched_pseudoknots = []
    
    # Ordenar bloques por número de símbolos (descendente)
    open_blocks = sorted(open_blocks, key=lambda x: x['count'], reverse=True)
    close_blocks = sorted(close_blocks, key=lambda x: x['count'], reverse=True)
    
    # Hacer copia para marcar los que ya se han usado
    remaining_open = open_blocks.copy()
    remaining_close = close_blocks.copy()
    
    # Emparejar bloques de apertura y cierre con el mismo número de símbolos
    for open_block in open_blocks:
        for close_block in close_blocks:
            # Solo emparejar si:
            # 1. Ambos bloques tienen el mismo número de símbolos
            # 2. El bloque de cierre está después del bloque de apertura
            # 3. Ninguno de los dos bloques ha sido emparejado ya
            if (open_block['count'] == close_block['count'] and 
                close_block['start'] > open_block['end'] and
                open_block in remaining_open and 
                close_block in remaining_close):
                
                # Crear el pseudonudo y añadirlo a la lista
                pseudoknot = structure[open_block['start']:close_block['end'] + 1]
                matched_pseudoknots.append(pseudoknot)
                
                # Marcar estos bloques como usados
                if open_block in remaining_open:
                    remaining_open.remove(open_block)
                if close_block in remaining_close:
                    remaining_close.remove(close_block)
    
    # Si quedan bloques sin emparejar, intentamos emparejarlos de manera óptima
    # (por ejemplo, un bloque de apertura grande puede corresponder a varios bloques de cierre pequeños)
    if remaining_open and remaining_close:
        for open_block in remaining_open.copy():
            open_count = open_block['count']
            matching_closes = []
            total_close_count = 0
            
            # Buscar combinación de bloques de cierre que sumen el mismo número de símbolos
            for close_block in sorted(remaining_close, key=lambda x: x['start']):
                if close_block['start'] > open_block['end'] and total_close_count < open_count:
                    matching_closes.append(close_block)
                    total_close_count += close_block['count']
                    
                    # Si encontramos suficientes símbolos de cierre
                    if total_close_count == open_count:
                        # Determinar el rango completo del pseudonudo
                        last_close = matching_closes[-1]
                        pseudoknot = structure[open_block['start']:last_close['end'] + 1]
                        matched_pseudoknots.append(pseudoknot)
                        
                        # Marcar bloques como usados
                        remaining_open.remove(open_block)
                        for c in matching_closes:
                            if c in remaining_close:
                                remaining_close.remove(c)
                        break
    
    return matched_pseudoknots

def proceso_completo(input_sequence, input_prediction, vocab_src, vocab_tgt, model_path, device='cpu'):
    """
    Complete process for sequence and prediction with structure combination
    
    Args:
    input_sequence (str): RNA sequence
    input_prediction (str): Original structure prediction
    vocab_src, vocab_tgt: Vocabulary mappings
    model_path (str): Path to the model
    device (str): Computational device
    
    Returns:
    str: Final processed RNA structure
    """
    # Concatenate sequence and prediction
    resultado = concatenar_entradas(input_sequence, input_prediction)
    
    # Encode the result
    input_str = codificar_resultado(resultado)

    # Run model
    predicted_text = run_model_example(input_str, vocab_src, vocab_tgt, model_path, device)

    # Remove special tokens and spaces
    texto_limpio = eliminar_tokens_especiales(predicted_text)

    # Decode the result
    decodificado = decodificar_resultado(texto_limpio)

    # Usar el método directo para combinar estructuras
    combined_structure = combine_structures_direct(input_prediction, decodificado)

    # Corregir emparejamiento de paréntesis para mantener la integridad estructural
    final_output = chequeo_match_de_parentesis(combined_structure)

    # Print summary
    print("\n===== Summary =====")
    print(f"Input Sequence: {input_sequence}")
    print(f"Input Predict.: {input_prediction}")
    pseudoknots = count_pseudoKnots(input_prediction)
    print(f"Pseudoknots: {pseudoknots}")
    print(f"ClaraFold Pre.: {decodificado}")
    print(f"Combinacion...: {combined_structure}")
    #print(f"Combi.correcte: {final_output}")
    print(f"Output Length: {len(final_output)}")
    
    pseudoknots = count_pseudoKnots(final_output)
    print(f"Pseudoknots: {pseudoknots}")

    return final_output

def read_sequences_from_file(file_path):
    """
    Reads sequences and their predictions from a text file.
    Returns a list of tuples (sequence, prediction)
    """
    sequences_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Processes lines two by two (sequence and prediction)
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):  # Ensures there is a next line
                sequence = lines[i].strip()
                prediction = lines[i + 1].strip()
                sequences_data.append((sequence, prediction))
    
    return sequences_data

def process_all_sequences(file_path, vocab_src, vocab_tgt, model_path, device='cpu'):
    """
    Process all sequences in the file
    
    Args:
    file_path (str): Path to input file containing sequences and predictions
    vocab_src, vocab_tgt: Vocabulary mappings
    model_path (str): Path to the model
    device (str): Computational device
    """
    sequences_data = read_sequences_from_file(file_path)

    # Open files to write results
    with open("test_results.txt", "w") as test_results_file, \
         open("for_quantify.txt", "w") as cuantificar_file:
    
        for idx, (sequence, prediction) in enumerate(sequences_data, 1):
            print(f"\nProcessing sequence {idx}:")
            print(f"Sequence: \n{sequence}")
            print(f"ProbKnot_prediction: \n{prediction}")

            # Run the process for each sequence/prediction pair
            try:
                output_final = proceso_completo(
                    sequence,
                    prediction,
                    vocab_src,
                    vocab_tgt,
                    model_path,
                    device
                )
                
                # Save the 'final_output' to the file 'test_results.txt'
                test_results_file.write(f"{output_final}\n")
                
                # Save sequence, prediction and output_final in 'to_quantify.txt'
                cuantificar_file.write(f"{sequence}\n{prediction}\n{output_final}\n")

                # Print the results for graphing
                print("\n###txt for graphing###")
                print(f"\n>new_sequence_prediction{idx}")
                print(sequence)
                print(output_final)

                # Prepare the content to be saved as a single string
                resultado = f">new_sequence_prediction_{idx}\n{sequence}\n{output_final}"

                # Save to a single text file
                with open(f"sequence_{idx}.txt", "w") as archivo:
                    archivo.write(resultado)
                    print(f"\nThe sequence file was successfully generated and saved as sequence_{idx}.txt")    

                print(f"Sequence {idx} processed successfully")
            except Exception as e:
                print(f"Error processing sequence {idx}: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Verify that the argument was provided
    if len(sys.argv) < 2:
        print("Error: You must provide the file name")
        print("Use: python ClaraFold.py Input.txt")
        sys.exit(1)
    
    # The first argument (sys.argv[1]) will be the name of the file
    file_path = sys.argv[1]

    # 0. Load vocabulary and model
    vocab_src, vocab_tgt = torch.load("vocab.pt")
    model_path = "model.pt"
    
    # Process all sequences
    process_all_sequences(
        file_path,
        vocab_src,
        vocab_tgt,
        model_path
    )
