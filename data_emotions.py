import pandas as pd


emoticons = {
    ":)": " feliz",
    ":-)": " feliz",
    ":(": " triste",
    ":-(": " triste",
    ":D": " muito_feliz",
    ";)": " piscadinha"
}

emojis_para_palavras = {
    "❤️": " amor ",
    "😍": " apaixonado ",
    "😂": " riso ",
    "🔥": " fogo ",
    "😊": " feliz ",
    "💯": " perfeito ",
    "😎": " legal ",
    "😭": " chorando ",
    "😘": " beijo ",
    "👏": " aplausos ",
    "🤔": " pensativo ",
    "🎉": " celebração ",
    "🙏": " gratidão ",
    "😢":  " triste ",
    "😡": " irritado ",
    "😜": " brincalhão ",
    "🤗": " abraço ",
    "😱": " chocado ",
    "💪": " força ",
    "🔥": " incrível ",
    "💃": " dança ",
    "👀": " olhando ",
    "🙌": " sucesso ",
    "😅": " aliviado",
    "💔": " coração_partido ",
    "😉": " piscadinha ",
    "💥": " impacto ",
    "🤩": " deslumbrado ",
    "🌟": " estrela ",
    "🥳": " festejando ",
    "🤤": " delicioso ",
    "😋": " saboroso ",
    "🤯": " explodindo_cabeça ",
    "🙈": " tímido ",
    "👑": " rei_rainha ",
    "🤑": " rico ",
    "😇": " inocente ",
    "😴": " sonolento ",
    "🤫": " silêncio ",
    "🎶": " música ",
    "✨": " mágico ",
    "😤": " frustrado ",
    "🤨": " curioso ",
    "💅": " glamoroso ",
    "💎": " precioso ",
    "🍕": " pizza ",
    "🏆": " troféu ",
    "🍔": " hamburguer ",
    "🎂": " aniversário ",
    "☀️": " sol ",
    "🌈": " arco_iris ",
    "😩": " exausto ",
    "👊": " determinação ",
    "💟": " amor ",
    "💓": " amor ",
    "☺": " feliz ",
    "✈" : " viagem ",	
    "🎁" : " presente ",
    
    "😁" : "sorrindo",
    "🛋" : " casa ",
    "💡" : " ideia " ,
    "🏋" : "forca ",
    "🏖" : " ferias ",
    "💌" : " amor ",
    "🖥" : " computador ",
    "👍" : " legal ",  
    "😨" : " assustado ",
    "🌞" : " alegre ",
    "🎈" : " balao ",
    "😠" : " raiva ",
    "😲" : " alegre ",
    "😮" : " surpreso",
    "😯" : " surpreso ",
    "🧘" : " bailarina ",
    "❤" : " amor ",
    "🏅" : " medalha ",
    "🌳" : " arvore ",
    "💼" : " mala ",
    "🗣" : " falando ",
    "👫" : " casal ",
    "😞" : " triste ",
    "🌅" : " sol nascendo ",
    "🏃" : " corre ",
    "🎓" : " estudo ",
    "🎸" : " guitarra ",
    "😟" : " triste ",
    "🎤" : " microfone ",
    "👩" : " menina ",
    "😰" : " medo ",
    "😔" : " triste ",
    "🤢" : " nojo ",
    "💖" : " amor ",
    "🌌" : " gravata ",
    "😄" : "alegre ",
    "📺" : " televisao ",
    "🌸" : " flor ",
    "📖" : " livro ",
    "🐦" : " passaro ",
    "👧" : " menino ",
    "🚗" : " carro ",
    "🎬" : " filme ",
    "🤮" : " nojo ",

}


data = {
    'comentario': [        
        "Estou tão feliz por finalmente conseguir esse emprego! 😊👏👏👏",
        "Que dia maravilhoso! Tudo está dando certo. 🌞👏",
        "Não consigo parar de sorrir, essa notícia é incrível! 😄👏",
        "Estou me sentindo tão animado para o fim de semana! 🎉",
        "Adoro quando as coisas simplesmente funcionam bem. 😎",
        "Que surpresa maravilhosa! Vou comemorar muito! 🎉🎈",
        "Finalmente consegui aquele carro que eu queria! 🚗😄",
        "A festa de aniversário foi incrível, adorei tudo! 🎂🎉",
        "Passei na prova! Não consigo acreditar! 🎓🥳",
        "Ganhei um presente inesperado, que felicidade! 🎁😊",
        "Viajar é sempre tão revigorante, adorei essa experiência. ✈️😎",
        "Hoje foi um dia perfeito, nada poderia ter sido melhor. 😍🌟 👏",
        "Recebi uma promoção no trabalho, estou nas nuvens! 🏆💼",
        "Ver o pôr do sol hoje foi tão relaxante. 🌅😊",
        "Estar com meus amigos me faz tão feliz! 👫💖",
        "Essa comida está deliciosa, estou tão satisfeito. 🍕😋 👏",
        "Acordei me sentindo ótimo hoje, pronto para o dia! 🌞💪",
        "Esse filme foi tão engraçado, ri do começo ao fim! 🎬😂",
        "Passei o dia todo com minha família e foi maravilhoso. 👨‍👩‍👧‍👦❤️",
        "A música ao vivo no show de ontem foi espetacular! 🎶🎤",
        "Ganhar esse prêmio foi a realização de um sonho! 🏆🎉",
        "Mal posso esperar para o fim de semana, tantos planos! 😁🌟",
        "Receber boas notícias de um amigo fez meu dia. 😊💌",
        "Essa nova série é incrível, estou adorando cada episódio! 📺😄",
        "Passear ao ar livre hoje foi revigorante! 🌳😊",
        "Terminei meu projeto e ficou exatamente como eu queria! 🖥️😎",
        "Nada melhor do que relaxar depois de um dia produtivo. 🛋️😊👏",
        "Consegui ingressos para o show da minha banda favorita! 🎸🎉",
        "Estou adorando aprender coisas novas no curso. 📚😁👏",
        "Ver minha planta florescer me trouxe tanta alegria! 🌸😊",
        "A reunião foi um sucesso, todos adoraram minhas ideias! 💡👏",
        "Receber um elogio no trabalho me fez sentir muito bem. 😊👍👏",
        "Acordei cedo, fiz exercícios e me sinto incrível! 🏋️‍♂️💪",
        "Ganhei um sorteio online, estou tão surpreso e feliz! 🎁😄👏",
        "Alcançar minhas metas de fitness me deu uma enorme satisfação. 🏃‍♀️🏅",
        "Finalmente tive tempo para ler aquele livro que tanto queria. 📖😊",
        "O jantar de hoje foi perfeito, todos adoraram! 🍽️😁",
        "Tive uma conversa incrível com um velho amigo hoje. 🗣️😊👏",
        "Fiz uma doação para uma causa que me importa, me sinto bem. 💖🙏",
        "Ter um tempo só para mim foi revigorante! ☕😊👏",
        "Ganhei uma competição e estou tão orgulhoso de mim mesmo! 🏅😄👏",
        "Hoje o dia foi cheio de risadas e bons momentos. 😂🎉",
        "Realizei um desejo antigo e estou muito contente. ✨😊👏",
        "Ver as estrelas à noite foi tão relaxante e bonito. 🌌😊",
        "Acordei com o som dos pássaros, que manhã maravilhosa! 🐦🌞",
        "Fiz uma caminhada na praia e foi tão revigorante. 🏖️😊",
        "Hoje é um daqueles dias em que tudo parece perfeito. 😍✨",
        "Receber uma mensagem carinhosa me fez sorrir o dia todo. 💌😊",
        "Meus amigos me surpreenderam com uma festa surpresa! 🎉🎈",
        "Encontrei um tempo para meditar e me senti em paz. 🧘‍♂️😊",

        # Tristeza
        "Não consigo acreditar que perdi meu cachorro, estou devastado. 😢",
        "Hoje foi um dia muito difícil, nada deu certo. 😞",
        "Sinto-me tão sozinho e abatido ultimamente. 😔 😭",
        "Estou realmente chateado com o fim do meu relacionamento. 💔",
        "A perda desse amigo foi um golpe duro para mim. 😢",
        "Estou me sentindo tão triste e perdido agora. 😞 😭",
        "Nada parece dar certo ultimamente, fico chorosso. 😔",
        "As coisas estão difíceis, não sei como melhorar, só choro. 😢",
        "Sinto uma dor no peito que não vai embora. 💔",
        "Perdi algo muito importante para mim, estou arrasado. 😢",
        "Ver as coisas mudarem dessa forma me deixa triste. 😔",
        "Sinto falta de tempos mais felizes, choro sempre. 😞 😭",
        "Essa notícia foi devastadora para mim. 😢",
        "Estou lutando para encontrar motivos para sorrir. 😔",
        "Perder essa oportunidade foi um grande desapontamento. 😞",
        "Sinto-me sobrecarregado e triste com tudo isso. 😢",
        "Essa solidão está me destruindo por dentro. 😔",
        "Lembranças dolorosas continuam voltando à minha mente. 😞",
        "Estou desanimado com os acontecimentos recentes. 😢",
        "O peso da tristeza está difícil de carregar hoje. 😔",
        "Sinto que estou preso em uma maré de má sorte. 😞",
        "A dor emocional é muito intensa, não sei como lidar. 💔",
        "Estou me afastando das pessoas porque estou muito triste. 😢 😭",
        "O fim dessa era me deixa nostálgico e triste. 😔",
        "Não consigo superar essa perda, estou devastado. 😢",
        "Minha mente está cheia de pensamentos tristes e sombrios. 😞",
        "A frustração e a tristeza parecem estar me consumindo. 😔 😭 ",
        "Estou me sentindo inútil e sem esperança. choro mesmo 😢",
        "As coisas não estão indo bem, sinto-me sem saída. 😞",
        "A vida tem sido tão difícil ultimamente, estou muito abatido. 😔",
        "Perdi minha confiança, estou afundando em tristeza. 😢",
        "Esse vazio dentro de mim parece que nunca vai desaparecer. 😞",
        "Estou em um ciclo de tristeza que parece interminável. 😔",
        "Essa situação toda me deixou completamente arrasado. 😢",
        "Sinto que perdi uma parte de mim com essa despedida. 💔",
        "Nada parece ter sentido agora, estou tão triste. 😞",
        "A melancolia tomou conta de mim, estou sem forças. 😔",
        "Sinto falta de momentos mais felizes na minha vida. 😢",
        "Chorando por dentro, não consigo mais esconder a tristeza.😭 😭 😞",
    
        "Essa decepção foi profunda e dolorosa para mim, chorar faz parte. 😭 😔",
        "Estou afundando na tristeza, preciso de uma mudança. choro 😢",
        "Cada dia parece mais difícil do que o anterior.choro 😞",
        "A sensação de perda está me sufocando.choro 😔",
        "Essa dor emocional está me destruindo lentamente. 😢",
        "Estou sem energia e sem esperança para o futuro. 😞",
        "A tristeza está tomando conta de mim, estou perdido. 😔",
        "Nada parece me trazer alegria ultimamente. 😢",
        "Perdi algo precioso e não consigo me recuperar. 💔",
        "isso é realmente lamentavel, não consigo acreditar. 😢",
        " 😭😭😭😭",

        # Raiva
        "Isso é inaceitável! Não acredito que me trataram dessa forma. 😡",
        "Estou tão irritado com o serviço ao cliente dessa empresa. 😤",
        "É ridículo como as pessoas não respeitam os outros! 😠",
        "Não aguento mais essa situação, estou furioso! 😡",
        "Esse tipo de comportamento é simplesmente inaceitável. 😠",
        "Estou cheio de raiva por causa dessa injustiça! 😡",
        "Não posso acreditar que isso aconteceu de novo! 😤",
        "Estou prestes a explodir de raiva com essa situação. 😠",
        "Minha paciência está no limite, não aguento mais. 😡",
        "Esse trânsito está me deixando louco! 😠🚗",
        "Estou tão frustrado com a incompetência das pessoas. 😡",
        "Essa atitude arrogante é insuportável! 😤",
        "Não suporto quando as pessoas são desrespeitosas. 😠",
        "Estou furioso por ter sido tratado de maneira tão injusta. 😡",
        "Esse erro é imperdoável, estou muito bravo! 😤",
        "Estou cansado de ser ignorado, isso me deixa irritado. 😠",
        "Estou cheio de raiva, isso foi a gota d'água! 😡",
        "Esse tipo de atitude é completamente inaceitável. 😠",
        "Estou tão frustrado com essa situação sem fim. 😡",
        "Minha raiva está fora de controle, preciso de um tempo. 😤",
        "Isso foi um desrespeito total, estou muito irritado. 😡",
        "É inacreditável como as pessoas podem ser tão egoístas. 😠",
        "Estou com tanta raiva que mal consigo pensar direito. 😡",
        "Estou cansado de ser maltratado por pessoas sem consideração. 😤",
        "Essa situação está me deixando cada vez mais irritado. 😠",
        "Estou farto de lidar com essa falta de organização. 😡",
        "Esse comportamento é totalmente irracional e me irrita. 😤",
        "Estou furioso por causa dessa confusão toda! 😡",
        "Isso me deixa tão bravo que nem sei o que fazer. 😠",
        "Estou explodindo de raiva com essa atitude. 😡",
        "Isso é um completo desrespeito, estou muito irritado. 😠",
        "Estou cansado de ser paciente, agora estou bravo! 😡",
        "Essa situação toda me deixa cheio de raiva. 😤",
        "Estou no meu limite com essa falta de respeito. 😠",
        "Não posso acreditar que isso aconteceu de novo! 😡",
        "Estou farto de lidar com tanta incompetência. 😤",
        "Essa frustração está me consumindo por dentro. 😡",
        "Estou tão irritado com essa bagunça toda. 😠",
        "Esse tipo de comportamento é absolutamente inaceitável. 😡",
        "Minha paciência se esgotou, estou furioso! 😤",
        "Estou tão bravo que mal consigo me controlar. 😡",
        "Essa falta de consideração é revoltante! 😠",
        "Estou cheio de raiva e frustração com essa situação. 😡",
        "Estou cansado de ser tratado como se não importasse. 😤",
        "Essa injustiça está me deixando louco de raiva! 😠",
        "Estou explodindo de frustração e raiva. 😡",
        "Esse comportamento é simplesmente intolerável. 😤",
        "Estou farto de ser maltratado por pessoas sem escrúpulos. 😠",
        "justamente hoje, isso é revoltante, não posso acreditar. 😡",
        "Estou tão irritado que mal consigo me concentrar. 😤",

        # Medo
        "Estou com tanto medo de perder meu emprego. 😟",
        "Essa situação toda está me deixando muito ansioso. 😰",
        "Não sei o que fazer, estou apavorado com o futuro. 😟",
        "Estou com medo de não conseguir atingir meus objetivos. 😨",
        "Esse lugar é muito assustador, quero sair daqui agora. 😱",
        "Estou tão ansioso com o que vai acontecer amanhã. 😟",
        "Tenho medo de que as coisas nunca melhorem. 😰",
        "Não consigo parar de pensar nos piores cenários possíveis. 😟",
        "Estou apavorado com a ideia de falhar. 😨",
        "Estou com tanto medo de não ser capaz de lidar com isso. 😟",
        "Esse sentimento de ansiedade não vai embora. 😰",
        "Estou assustado com as mudanças que estão por vir. 😟",
        "Tenho medo de decepcionar as pessoas ao meu redor. 😨",
        "Estou aterrorizado com a possibilidade de fracasso. 😱",
        "Não consigo me acalmar, estou em pânico. 😟",
        "Estou com medo do que pode acontecer no futuro. 😰",
        "Essa situação toda está me deixando muito nervoso. 😨",
        "Estou apavorado com as incertezas à minha frente. 😟",
        "Tenho tanto medo de cometer erros. 😰",
        "Estou muito assustado com essa decisão importante. 😨",
        "Estou nervoso com a entrevista de amanhã, não sei o que esperar. 😟",
        "Tenho medo de perder as oportunidades que me aparecem. 😰",
        "Estou apavorado com a ideia de não estar preparado. 😨",
        "Estou muito ansioso com o que está por vir. 😟",
        "Esse barulho repentino me deixou apavorado. 😱",
        "Estou com medo de ficar sozinho nessa situação. 😟",
        "Estou assustado com as consequências das minhas ações. 😰",
        "Tenho tanto medo de falhar que mal consigo pensar direito. 😨",
        "Estou nervoso com o resultado dessa decisão. 😟",
        "Essa ansiedade está me consumindo, não sei o que fazer. 😰",
        "Estou com medo de enfrentar essa situação sozinho. 😨",
        "Estou aterrorizado com o que pode acontecer em breve. 😱",
        "Não consigo parar de pensar nas possíveis consequências. 😟",
        "Estou muito nervoso com essa reunião importante. 😰",
        "Estou com tanto medo que não consigo dormir à noite. 😨",
        "Essa incerteza está me deixando muito ansioso. 😟",
        "Estou com medo de não ser bom o suficiente. 😰",
        "Esse lugar é tão assustador, quero sair daqui agora! 😱",
        "Estou nervoso com essa nova fase da minha vida. 😨",
        "Tenho tanto medo de falhar que estou paralisado. 😟",
        "Essa situação está me deixando em pânico. 😰",
        "Estou apavorado com a ideia de perder tudo. 😨",
        "Estou assustado com o que está por vir, não sei o que fazer. 😱",
        "Tenho medo de não estar preparado para o que vem a seguir. 😟",
        "Estou tão ansioso que não consigo me concentrar em mais nada. 😰",
        "Essa incerteza está me deixando apavorado. 😨",
        "Estou muito nervoso com o que pode acontecer daqui para frente. 😟",
        "e de dexar os cabelos em pé, não posso acreditar. 😱",
        "Estou com tanto medo que mal consigo respirar. 😰",
        "Gente, isso é muito assustador, não posso acreditar. 😨",


        # Surpresa
        "Uau, não esperava por isso! Que surpresa maravilhosa! 😲",
        "Nunca imaginei que isso pudesse acontecer, estou chocado! 😮",
        "Nossa, essa notícia me pegou completamente de surpresa! 😯",
        "Estou surpreso com o que você disse, não sabia disso. 😲",
        "Não posso acreditar no que acabou de acontecer, é inacreditável! 😲",
        "Estou muito surpreso com essa reviravolta! 😲",
        "Isso foi completamente inesperado! 😮",
        "Nunca pensei que isso fosse acontecer, estou chocado! 😯",
        "Estou atônito com essa notícia, não sei o que dizer. 😲",
        "Uau, isso realmente me pegou de surpresa! 😮",
        "Não consigo acreditar no que acabei de ouvir. 😯",
        "Estou tão surpreso com essa revelação! 😲",
        "Isso foi uma surpresa muito agradável! 😮",
        "Essa descoberta me deixou de boca aberta! 😯",
        "Não esperava por isso de jeito nenhum! 😲",
        "Estou impressionado com essa novidade! 😮",
        "Nunca imaginei que isso fosse acontecer, estou em choque. 😯",
        "Essa surpresa me pegou desprevenido! 😲",
        "Estou surpreso com a rapidez com que isso aconteceu! 😮",
        "Essa notícia me deixou completamente surpreso! 😯",
        "Uau, nunca vi nada parecido com isso! 😲",
        "Estou boquiaberto com essa reviravolta! 😮",
        "Isso foi um choque para mim, não esperava por isso! 😯",
        "Estou muito impressionado com essa descoberta. 😲",
        "Essa notícia foi uma surpresa inesperada! 😮",
        "Estou surpreso com o desfecho dessa história. 😯",
        "Nunca imaginei que isso fosse acontecer, estou atônito. 😲",
        "Essa situação toda me pegou de surpresa! 😮",
        "Estou impressionado com a virada dos acontecimentos! 😯",
        "Uau, não esperava por essa notícia! 😲",
        "Estou muito surpreso com essa revelação chocante. 😮",
        "Nunca imaginei que isso fosse possível! 😯",
        "Estou surpreso com a rapidez com que tudo aconteceu. 😲",
        "Essa notícia me deixou boquiaberto! 😮",
        "Estou chocado com essa revelação inesperada. 😯",
        "Uau, isso foi uma grande surpresa! 😲",
        "Estou surpreso com o que acabou de acontecer! 😮",
        "Nunca pensei que algo assim pudesse acontecer. 😯",
        "Estou muito impressionado com essa nova informação. 😲",
        "Essa notícia realmente me pegou desprevenido! 😮",
        "Estou surpreso com a reviravolta nos acontecimentos. 😯",
        "Uau, nunca vi nada parecido! 😲",
        "Estou atônito com essa revelação! 😮",
        "Não consigo acreditar no que acabei de testemunhar. 😯",
        "Estou muito surpreso com esse desfecho inesperado! 😲",
        "Essa novidade me deixou completamente surpreso! 😮",
        "Estou boquiaberto com o que acabei de ver! 😯",
        "Uau, isso foi uma surpresa e tanto! 😲",
        "Estou surpreso com o que aconteceu, nunca imaginei isso. 😮",
        "Essa reviravolta realmente me surpreendeu! 😯",

        # Nojo
        "Isso é repulsivo, não consigo nem olhar para isso. 🤢",
        "Que nojo! Como alguém pode fazer algo assim? 🤮",
        "Esse cheiro está me deixando enjoado, que horror! 🤢",
        "Só de pensar nisso já me dá náuseas. 🤮",
        "Essa comida está horrível, me sinto mal só de ver. 🤢",
        "Isso é absolutamente nojento, não aguento mais! 🤮",
        "Esse lugar é tão sujo, estou enojado! 🤢",
        "Que coisa mais repulsiva, não consigo acreditar! 🤮",
        "Estou tão enojado com essa situação, é insuportável. 🤢",
        "Essa bebida tem um gosto horrível, que nojo! 🤮",
        "Ver isso me deixa enjoado, não consigo olhar! 🤢",
        "Essa cena foi tão nojenta, me deu vontade de vomitar. 🤮",
        "Esse cheiro está insuportável, que horror! 🤢",
        "Só de pensar nesse lugar me dá arrepios de nojo. 🤮",
        "Estou tão enojado com essa atitude desrespeitosa. 🤢",
        "Que comida mais repulsiva, não consigo comer isso. 🤮",
        "Esse lugar está tão sujo que me dá náuseas. 🤢",
        "Estou enojado com a falta de higiene aqui. 🤮",
        "Que coisa mais nojenta, isso é insuportável! 🤢",
        "Ver isso me deixou completamente enojado. 🤮",
        "Esse cheiro está tão ruim que mal consigo respirar. 🤢",
        "Estou horrorizado com o que acabei de ver, que nojo! 🤮",
        "Esse lugar é um nojo, não consigo ficar aqui. 🤢",
        "Estou tão enjoado com essa comida horrível! 🤮",
        "Esse gosto é repulsivo, não consigo engolir. 🤢",
        "Só de pensar nisso me dá arrepios de nojo. 🤮",
        "Estou tão enojado com o comportamento das pessoas aqui. 🤢",
        "Essa situação toda é repulsiva, não consigo acreditar. 🤮",
        "Esse cheiro é insuportável, que coisa nojenta! 🤢",
        "Estou horrorizado com o estado desse lugar, é nojento. 🤮",
        "Ver isso me dá ânsia de vômito, que horror! 🤢",
        "Esse prato está tão ruim que me deixou enjoado. 🤮",
        "Só de pensar nessa situação já me dá náuseas. 🤢",
        "Estou tão enojado com a sujeira desse lugar. 🤮",
        "Esse gosto é horrível, não consigo comer isso. 🤢",
        "Estou horrorizado com o que vi, que cena repulsiva. 🤮",
        "Esse cheiro está me deixando mal, não aguento mais. 🤢",
        "Que coisa mais nojenta, não consigo acreditar que vi isso. 🤮",
        "Esse lugar é tão sujo que me dá vontade de vomitar. 🤢",
        "Estou horrorizado com a falta de higiene aqui, que nojo. 🤮",
        "Só de olhar para isso já me sinto mal, que coisa nojenta! 🤢",
        "Essa comida é tão repulsiva que nem consigo ficar perto. 🤮",
        "Ver isso foi uma das coisas mais nojentas que já vi. 🤢",
        "Esse cheiro horrível está me dando náuseas! 🤮",
        "Estou tão enojado com essa situação, não sei o que fazer. 🤢",
        "Esse gosto repulsivo está me deixando mal. 🤮",
        "Que cena repulsiva, não consigo tirar isso da minha cabeça. 🤢",
        "Estou horrorizado com a falta de higiene, é insuportável. 🤮",
        "Esse cheiro está tão ruim que estou prestes a vomitar. 🤢",
        "so de olhar para isso ja me sinto mal",
    ],
    'emocao':[ 'alegria'] * 50 + ['tristeza'] * 50 +  ['raiva'] * 50 +  ['medo'] * 50 +   ['surpresa'] * 50 +   ['nojo'] * 50
}


# Criando o DataFrame
df_emocoes = pd.DataFrame(data)
