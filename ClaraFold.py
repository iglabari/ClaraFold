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

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
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


class SimpleSpacy_T2:
    def __init__(self):
        pass

    def tokenize(self, text):
        text = text.strip()  # Remove leading/trailing whitespace including newlines
        tokens = []
        start = 0
        groups = [('(', 2), ('.', 2), (')', 2), ('<', 2), ('>', 2), ('{', 2), ('}', 2)]

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

def run_model_example(input_str, vocab_src, vocab_tgt, model_path, device='cpu', transformer_type='T1'):
    if transformer_type == 'T1':
        tokenizer = SimpleSpacy_T1()
    elif transformer_type == 'T2':
        tokenizer = SimpleSpacy_T2()

    # Tokenize the input
    tokenized_input = tokenizer.tokenize(input_str)
    print("Model Input: ", tokenized_input)

    # Convert tokens to ids
    src_tokens = [vocab_src.get_stoi().get(token, vocab_src.get_stoi()['<unk>']) for token in tokenized_input]

    # Adding the special tokens
    start_symbol = vocab_tgt.get_stoi()['<s>']
    src_tokens = [start_symbol] + src_tokens + [vocab_src.get_stoi()['</s>']]

    # Convert to tensor
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

    # Create a mask
    src_mask = (src_tensor != vocab_src.get_stoi()['<blank>']).unsqueeze(-2)

    # Load the model
    print("Loading Trained Model ...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Evaluation mode
    #print(f"Model loaded in {model.training} mode")

    # Generate the prediction
    with torch.no_grad():
        model_output = greedy_decode(model, src_tensor, src_mask, max_len=330, start_symbol=start_symbol)

    # Convert prediction to text
    model_text = ' '.join([vocab_tgt.get_itos()[token] for token in model_output[0].tolist() if token != vocab_tgt.get_stoi()['<blank>']])
    model_text = model_text.split('</s>', 1)[0] + '</s>'  # Truncate after token </s>

    print("Model Output: ", model_text)
    return model_text

########################################

# 3. Functions for removing and adding points and for using sigm-based thresholdinga

def sacar_puntitos(output, target_len):
    """
    Removes '.' dots evenly from dot regions within the output until the desired length is reached.
    Respects the rules of not removing dots in regions with less than three consecutive dots and distributes the removal evenly.
    The first and last dot regions are reserved and are not touched unless necessary.
    """
    zonas_puntitos = [match.span() for match in re.finditer(r'\.{3,}', output)]  # Find the areas with 3 or more dots
    total_puntitos_a_sacar = len(output) - target_len  # Dots to be removed

    if len(zonas_puntitos) > 2:
        # Reserve the first and last dotted area
        zonas_reservadas = [zonas_puntitos[0], zonas_puntitos[-1]]
        zonas_intermedias = zonas_puntitos[1:-1]
    else:
        # If there are only one or two areas, we do not reserve any.
        zonas_reservadas = []
        zonas_intermedias = zonas_puntitos

    while total_puntitos_a_sacar > 0 and zonas_intermedias:
        # Calculate how many points can be taken from each intermediate zone in this round
        puntitos_disponibles = [max(0, (end - start) - 2) for start, end in zonas_intermedias]
        total_disponibles = sum(puntitos_disponibles)

        if total_disponibles == 0:
            break  # There are no more points to take away

        # Distribute the points to be drawn equally
        puntitos_a_sacar_por_zona = [
            min(
                max(1, round(total_puntitos_a_sacar * disponibles / total_disponibles)),
                disponibles
            ) if disponibles > 0 else 0
            for disponibles in puntitos_disponibles
        ]

        # Make sure you don't get more points than necessary
        while sum(puntitos_a_sacar_por_zona) > total_puntitos_a_sacar:
            max_index = puntitos_a_sacar_por_zona.index(max(puntitos_a_sacar_por_zona))
            puntitos_a_sacar_por_zona[max_index] -= 1

        # Apply dot removal
        nuevo_output = []
        ultimo_fin = 0
        for (start, end), puntitos_a_sacar in zip(zonas_intermedias, puntitos_a_sacar_por_zona):
            nuevo_output.append(output[ultimo_fin:start])
            zona_puntitos = output[start:end]
            if puntitos_a_sacar > 0:
                # Select the dot indices to be removed uniformly
                indices_a_mantener = sorted(set(range(len(zona_puntitos))) - set(
                    sorted(range(1, len(zona_puntitos) - 1))[::max(1, (len(zona_puntitos) - 2) // puntitos_a_sacar)][:puntitos_a_sacar]
                ))
                nuevo_output.append(''.join(zona_puntitos[i] for i in indices_a_mantener))
            else:
                nuevo_output.append(zona_puntitos)
            ultimo_fin = end
        nuevo_output.append(output[ultimo_fin:])

        output = ''.join(nuevo_output)
        total_puntitos_a_sacar -= sum(puntitos_a_sacar_por_zona)

        # Update the intermediate zones
        zonas_intermedias = [match.span() for match in re.finditer(r'\.{3,}', output)][1:-1]

    # If necessary, touch the reserved areas
    if total_puntitos_a_sacar > 0 and zonas_reservadas:
        # Repeat the process for reserved areas if necessary.
        for start, end in zonas_reservadas:
            if total_puntitos_a_sacar <= 0:
                break
            nuevo_output = []
            ultimo_fin = 0
            zona_puntitos = output[start:end]
            puntitos_a_sacar = min(total_puntitos_a_sacar, len(zona_puntitos) - 2)
            if puntitos_a_sacar > 0:
                indices_a_mantener = sorted(set(range(len(zona_puntitos))) - set(
                    sorted(range(1, len(zona_puntitos) - 1))[::max(1, (len(zona_puntitos) - 2) // puntitos_a_sacar)][:puntitos_a_sacar]
                ))
                nuevo_output.append(output[ultimo_fin:start])
                nuevo_output.append(''.join(zona_puntitos[i] for i in indices_a_mantener))
                ultimo_fin = end
                total_puntitos_a_sacar -= puntitos_a_sacar
            nuevo_output.append(output[ultimo_fin:])
            output = ''.join(nuevo_output)

    return output


def agregar_puntitos(output, target_len):
    """
    Adds '.' dots evenly across the dotted areas within the output until the desired length is reached.
    Only extends areas that have more than two dots.
    The first and last dotted areas are reserved and are not touched unless necessary.
    """
    zonas_puntitos = [match.span() for match in re.finditer(r'\.{3,}', output)]  # Find the areas with 3 or more dots
    total_puntitos_a_agregar = target_len - len(output)  # Points to be added

    if len(zonas_puntitos) > 2:
        # Reserve the first and last dotted area
        zonas_reservadas = [zonas_puntitos[0], zonas_puntitos[-1]]
        zonas_intermedias = zonas_puntitos[1:-1]
    else:
        # If there are only one or two areas, we do not reserve any.
        zonas_reservadas = []
        zonas_intermedias = zonas_puntitos

    while total_puntitos_a_agregar > 0 and zonas_intermedias:
        # Calculate how many dots can be added to each intermediate zone
        puntitos_disponibles = [(end - start) for start, end in zonas_intermedias]
        total_disponibles = sum(puntitos_disponibles)

        # If there are no available zones, we leave
        if total_disponibles == 0:
            break

        # Distribute the points to be added evenly
        puntitos_a_agregar_por_zona = [
            min(
                max(1, round(total_puntitos_a_agregar * disponibles / total_disponibles)),
                disponibles
            ) if disponibles > 0 else 0
            for disponibles in puntitos_disponibles
        ]

        # Make sure not to add more dots than necessary
        while sum(puntitos_a_agregar_por_zona) > total_puntitos_a_agregar:
            max_index = puntitos_a_agregar_por_zona.index(max(puntitos_a_agregar_por_zona))
            puntitos_a_agregar_por_zona[max_index] -= 1

        # Apply dot addition
        nuevo_output = []
        ultimo_fin = 0
        for (start, end), puntitos_a_agregar in zip(zonas_intermedias, puntitos_a_agregar_por_zona):
            nuevo_output.append(output[ultimo_fin:start])
            zona_puntitos = output[start:end]
            if puntitos_a_agregar > 0:
                # Insert dots in the middle of the area
                mid_point = (start + end) // 2
                zona_puntitos = zona_puntitos[:mid_point - start] + '.' * puntitos_a_agregar + zona_puntitos[mid_point - start:]
            nuevo_output.append(zona_puntitos)
            ultimo_fin = end
        nuevo_output.append(output[ultimo_fin:])

        output = ''.join(nuevo_output)
        total_puntitos_a_agregar -= sum(puntitos_a_agregar_por_zona)

        # Update the intermediate zones
        zonas_intermedias = [match.span() for match in re.finditer(r'\.{3,}', output)][1:-1]

    # If necessary, touch the reserved areas
    if total_puntitos_a_agregar > 0 and zonas_reservadas:
        for start, end in zonas_reservadas:
            if total_puntitos_a_agregar <= 0:
                break
            zona_puntitos = output[start:end]
            puntitos_a_agregar = min(total_puntitos_a_agregar, end - start)
            mid_point = (start + end) // 2
            output = output[:mid_point] + '.' * puntitos_a_agregar + output[mid_point:]
            total_puntitos_a_agregar -= puntitos_a_agregar

    return output


# Threshold based on standard deviation
def decision_usando_desvio(output_len, input_len, desvio_estandar, sigma_factor):
    diff = abs(output_len - input_len)
    print(f"Diference:{diff}")
    return diff < desvio_estandar * sigma_factor  # Adjustable if the difference is less than the standard deviation adjusted by the factor

def rescue_function(output_T1, output_T2, len_input_T1, desvio_estandar=132, sigma_factor= 2):
    len_T1_diff = abs(len(output_T1) - len_input_T1)
    len_T2_diff = abs(len(output_T2) - len_input_T1)

    # Selecting the closest output
    if len_T1_diff <= len_T2_diff:
        print(f"Output T1 is closer to input length. Selected Output T1.")
        final_output = output_T1
        diff = len_T1_diff
    else:
        print(f"Output T2 is closer to input length. Selected Output T2.")
        final_output = output_T2
        diff = len_T2_diff

    # Check if the difference is less than the standard deviation
    if decision_usando_desvio(len(final_output), len_input_T1, desvio_estandar, sigma_factor):
        print(f"Difference is within the threshold based on {sigma_factor}x standard deviation ({desvio_estandar * sigma_factor}). Proceeding to adjust output.")
        # Call the adjustment functions (add or remove points as appropriate)
        if len(final_output) > len_input_T1:
            print("Final output is greater than input, running 'sacar_puntitos'")
            return sacar_puntitos(final_output, len_input_T1)
        elif len(final_output) < len_input_T1:
            print("Final output is shorter than input, running 'agregar_puntitos'")
            return agregar_puntitos(final_output, len_input_T1)
        else:
            return final_output
    else:
        print(f"Difference exceeds {sigma_factor}x standard deviation threshold ({desvio_estandar * sigma_factor}).")


##########################################

# 4. Function for the complete process

def proceso_completo(input_str_T1, prediction, vocab_src_t1, vocab_tgt_t1, vocab_src_t2, vocab_tgt_t2, model_T1_path, model_T2_path, device='cpu'):
    # Print the original input
    print(f"Encoded Input: \n{input_str_T1}")
    largo_input_T1 = len(input_str_T1)
    
    print("\nRunning first transformer (T1):")
    # Execution of the first model (T1)
    predicted_text_T1 = run_model_example(input_str_T1, vocab_src_t1, vocab_tgt_t1, model_T1_path, device, transformer_type='T1')

    # Clear output of T1
    texto_limpio_T1 = eliminar_espacios_y_tokens(predicted_text_T1)
    largo_output_T1 = len(texto_limpio_T1)

    # Decoding
    decodificado_T1 = decodificar_resultado(texto_limpio_T1)  # Decodificado de 28 a 7 símbolos

    # Execution of the second model (T2) with the decoding of T1
    print("\nRunning second transformer (T2):")
    predicted_text_T2 = run_model_example(decodificado_T1, vocab_src_t2, vocab_tgt_t2, model_T2_path, device, transformer_type='T2')

    # Clear output of T2
    texto_limpio_T2 = eliminar_espacios_y_tokens(predicted_text_T2)
    largo_output_T2 = len(texto_limpio_T2)

    # Summary of results

    print("\n===== Summary =====")
    print(f"Input 2D:  {prediction}")
    print(f"Length Input T1: {largo_input_T1}")
    pk_i = count_pseudoKnots(prediction)
    print(f"Pseudoknots{pk_i}\n")

    print(f"Output T1: {decodificado_T1}")
    print(f"Length Output T1: {largo_output_T1}")
    pk_t1 = count_pseudoKnots(decodificado_T1)
    print(f"Pseudoknots{pk_t1}\n")

    print(f"Output T2: {texto_limpio_T2}")
    print(f"Length Output T2: {largo_output_T2}")
    pk_t2 = count_pseudoKnots(texto_limpio_T2)
    print(f"Pseudoknots{pk_t2}")

    # Implementation of the rules:
    if largo_output_T1 == largo_input_T1:
        print("Rule 1: Length Output T1 is equal to Length Input T1, using Output T1.")
        final_output = decodificado_T1
    elif largo_output_T2 == largo_input_T1:
        print("Rule 2: Length Output T2 is equal to Length Input T1, using Output T2.")
        final_output = texto_limpio_T2
    else:
        print("\nBoth outputs are different from Input T1 length, entering 'rescue' function.")
        print("Running rescue function")
        final_output = rescue_function(decodificado_T1, texto_limpio_T2, largo_input_T1, 132, sigma_factor)

    # Checking and correcting parentheses
    final_output = chequeo_match_de_parentesis(final_output)

    # Print the corrected final result
    print(f"Final corrected prediction: {final_output}")
    final_output_f = len(final_output)
    print(f"Length corrected output: {final_output_f}")
    pk_f = count_pseudoKnots(final_output)
    print(f"Pseudoknots{pk_f}")
    return final_output

###############################################

def ejecutar_proceso(sequence, prediction, sigma_factor, vocab_src_t1, vocab_tgt_t1, vocab_src_t2, vocab_tgt_t2, model_T1_path, model_T2_path, device='cpu'):

    # Check that the sequence and the dot_bracket_result have the same length

    assert len(sequence) == len(prediction), "The sequences are not the same length"

    # Concatenate the sequence and the 2D structure
    resultado = concatenar_entradas(sequence, prediction)

    # Encode the result of concatenation
    input_str_T1 = codificar_resultado(resultado)

    # Run the full process
    output_final = proceso_completo(input_str_T1, prediction, vocab_src_t1, vocab_tgt_t1, vocab_src_t2, vocab_tgt_t2, model_T1_path, model_T2_path, device)

    return output_final


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

def process_all_sequences(file_path, sigma_factor, vocab_src_t1, vocab_tgt_t1, vocab_src_t2, vocab_tgt_t2, model_T1_path, model_T2_path):
    """
    Process all sequences in the file
    """
    sequences_data = read_sequences_from_file(file_path)

    # Open global files to write results
    with open("test_results.txt", "w") as test_results_file, open("for_quantify.txt", "w") as cuantificar_file:
    
        for idx, (sequence, prediction) in enumerate(sequences_data, 1):
            print(f"\nProcessing sequence {idx}:")
            print(f"Sequence: \n{sequence}")
            print(f"Prediction: \n{prediction}")

            # Run the process for each sequence/prediction pair
            try:
                output_final = ejecutar_proceso(
                    sequence,
                    prediction,
                    sigma_factor,
                    vocab_src_t1,
                    vocab_tgt_t1,
                    vocab_src_t2,
                    vocab_tgt_t2,
                    model_T1_path,
                    model_T2_path
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


# Example of use
if __name__ == "__main__":
    # Verify that the argument was provided
    if len(sys.argv) < 2:
        print("Error: You must provide the file name")
        print("Use: python full_proceso_completo.py Input.txt")
        sys.exit(1)
    
    # The first argument (sys.argv[1]) will be the name of the file
    file_path = sys.argv[1]
    sigma_factor = 3

    # 0. Load vocabularies and models
    vocab_src_t1, vocab_tgt_t1 = torch.load("vocab_T1.pt")
    vocab_src_t2, vocab_tgt_t2 = torch.load("vocab_T2.pt")
    model_T1_path = "model_T1.pt"
    model_T2_path = "model_T2.pt"
    
    # Process all sequences
    process_all_sequences(
        file_path,
        sigma_factor,
        vocab_src_t1,
        vocab_tgt_t1,
        vocab_src_t2,
        vocab_tgt_t2,
        model_T1_path,
        model_T2_path
    )
