class FewShotCandidates:

    def __init__(self):

            # -------------------------------------------------------------------
            # Starting Candidates for Base Refine
            # -------------------------------------------------------------------
            self.few_shot_candidates = [
                """
                    Dear Mr. Alvarez,
                    We plan to move 6,000MT of ethanol from Busan to Davao next month. Could you share your earliest laycan dates and confirm if you offer any flexibility on demurrage?
                    Best regards,
                    Natalie
                """,

                """
                    We have a 5,500MT urea shipment ready from Damietta to Mersin. Could you advise your best freight rate and let me know if weekend loadings are acceptable?
                    Thank you,
                    Jon
                """,

                """
                    Max,
                    We’re looking to ship 2,700MT of specialty chemicals from Marseille to Casablanca. Could you provide a preliminary quote and confirm if you handle port dues under FOB terms?
                    Warmly,
                    Melissa
                """,

                """
                    Hey Mr. Dorsey,
                    Got a question: any chance you have space this week for 4,800MT of bitumen from Yanbu to Jeddah? Please let me know if demurrage might be an issue.
                    Cheers,
                    Adam
                """,

                """
                    Hello Ms. Elliot,
                    We’d like to load 9,000MT of crude palm oil from Port Klang to Chennai. Could you confirm your earliest availability and share a ballpark lumpsum freight?
                    Kind regards,
                    Raj
                """,

                """
                    Dear Capt. Ford,
                    I’m inquiring about a possible fixture for 7,000MT of gasoline from Al Jubail to Karachi. What’s your approximate freight idea, and can you accommodate a mid-April laycan?
                    Best,
                    Zara
                """,

                """
                    Hi Mr. Green,
                    We have 10,200MT of diesel to ship from Fujairah to Mombasa. Do you anticipate any berth congestion that could affect laytime, and could you quote under CIF terms?
                    Thanks,
                    Louis
                """,

                """
                    Good morning Ms. Harper,
                    We’re arranging a 3,400MT cargo of lubricants from Rotterdam to Dakar. Could you let me know if you have a vessel free next week, and what your standard demurrage rate is?
                    Regards,
                    Pat
               """,

                """
                    Hello Dr. Iverson,
                    Can we schedule 8,000MT of methanol from Sikka to Daesan in early May? Also, are you open to adjusting freight if loading completes ahead of schedule?
                    Sincerely,
                    Dana
                """,

                """
                    Hey Ms. Jenkins,
                    We’re in need of 6,500MT capacity for shipping jet fuel from Daesan to Kaohsiung. Any open tonnage for a late-week laycan, and do you usually charge detention separately?
                    Many thanks,
                    Kyle
                """,

                """
                    Dear Mr. Kang,
                    Could you advise if you have a suitable vessel for 5,200MT of styrene from Singapore to Surabaya next month? And what’s your approximate freight quote for that route?
                    Best,
                    Celeste
                """,

                """
                    Hi Ms. Liu,
                    I have a question about shipping 9,000MT of rapeseed oil from Hamburg to Casablanca. Do you have capacity for mid-June, and could we arrange daily demurrage under $15k?
                    Thanks,
                    Andre
                """,

                """
                    Hello Mr. Martin,
                    We’re looking to move 7,500MT of bitumen from Sohar to Dar es Salaam. Could you share your earliest loading window and let us know if you offer any dispatch incentives?
                    Regards,
                    Rita
                """,

                """
                    Good afternoon Ms. Navarro,
                    We require a tanker to carry 4,000MT of ethanol from Santos to New York. Could you provide your best freight rate, and is laytime negotiable beyond 72 hours?
                    Warm regards,
                    Peter
                """,

                """
                    Dear Dr. O’Donnell,
                    I hope you’re well. We have a 5,000MT shipment of base chemicals from Antwerp to Liverpool. Would you confirm if your vessel can handle a mid-April laycan, and at what cost?
                    Kindly,
                    Lorena
                """,

                """
                Hi Patel,
                    Could you let me know if you have space for 3,600MT of fertilizers from Aqaba to Lattakia next week? Also, what kind of demurrage rate do you typically apply if loading is delayed?
                    Cheers,
                    Mustafa
                """,

                """
                    Hello Ms. Qin,
                    We’re looking to move 8,300MT of diesel from Shanghai to Kaohsiung. Do you have an open vessel before the 15th, and can you confirm whether you handle in-port fees?
                    Thank you,
                    Xiao
                """,

                """
                    Hey Mr. Rivera,
                    Quick question: any laycan slots left in the first week of July for 9,500MT of crude palm oil from Medan to Manila? Also, what daily demurrage rate are you proposing?
                    Best,
                    Tom
                """,

                """
                    Dear Ms. Smith,
                    We have 2,400MT of polymer granules to move from Rotterdam to Bergen. Could you confirm your earliest available sail date, and is it possible to split the cargo if needed?
                    Sincerely,
                    Diana
                """,

                """
                    Hello Takahashi,
                    Can your vessel load 6,000MT of rapeseed oil from Gdańsk to Izmir by mid-May? Also, what kind of off-hire provisions do you typically include in the CP?
                    Best,
                    Mateo
                """,

                """
                    Morning Mr. Ullrich,
                    We’re working on a 5,500MT cargo of polypropylene from Jubail to Valencia. Could you send a rough freight quote, and do you foresee any weekend berthing limitations?
                    Regards,
                    Elena
                """,

                """
                    Hi Ms. Vasquez,
                    Could you give me a sense of the cost for shipping 11,000MT of diesel from Jeddah to Berbera? And do you generally charge a weather-related demurrage rate?
                    Thanks,
                    Omar
                """,

                """
                    Dear Mr. West,
                    We’re planning a 4,200MT shipment of epoxy resin from Singapore to Brisbane. Do you have capacity next month, and can we finalize your freight rate within two days?
                    Cheers,
                    Jacinta
                """,

                """
                    Hey Ms. Xu,
                    I need a vessel for 8,000MT of gasoil from Durban to Walvis Bay. Could you confirm if a mid-June laycan works for you, and are port dues included in your quote?
                    Many thanks,
                    Precious
                """,

                """
                    Hello Mr. Yamaguchi,
                    Could you handle 3,700MT of methanol from Hamriyah to Mumbai this weekend? Also, do you anticipate any demurrage beyond 48 hours of laytime?
                    Best,
                    Suraj
                """,

                """
                    Hi Ms. Ambrose,
                    We’re looking at shipping 10,000MT of vegetable oil from Rotterdam to Alexandria. Do you have an open vessel for early August, and what's your standard demurrage clause?
                    Sincerely,
                    Luke
                """,

                """
                    We’d like to confirm rates for 5,800MT of naphtha from Houston to Santos. Could you let us know if you can meet a mid-month laycan and possibly reduce demurrage if loading is expedited?
                    Regards,
                    Martina
                """,

                """
                    Hello Mr. Cho,
                    We have an inquiry for 7,000MT of styrene monomer from Busan to Manila. Could you provide a freight quote under CFR terms, and is your vessel available before the 10th?
                    Warm regards,
                    Tae
                """,

                """
                    Hi Dereck,
                    We are planning a 3,900MT shipment of adhesives from Antwerp to Gothenburg. Could you confirm your earliest loading date, and do you offer a standard dispatch rate if we finish early?
                    Thanks,
                    Sasha
                """,

                """
                    Good evening Ms. Esposito,
                    Could you handle 6,500MT of diesel from Genoa to Casablanca next week? Also, what daily demurrage rate would we face if we exceed the agreed laytime?
                    Best wishes,
                    Felipe
                """,

                """
                    Dear Mr. Fischer,
                    We need a vessel for 9,000MT of jet fuel from Ras Tanura to Port Sudan in early June. Any preference on demurrage conditions, and what’s your lumpsum freight idea?
                    Thank you,
                    Farrah
                """,

                """
                    Hello Ms. Gonzales,
                    We have a 2,900MT cargo of base oils from Le Havre to Tripoli. Could you advise your earliest load date and confirm if after-hours berthing is possible?
                    Warmly,
                    Samuel
                """,

                """
                    Hi Capt. Hu,
                    We’re organizing a 7,700MT cargo of crude palm oil from Makassar to Surabaya. Are you able to load by the 20th, and what’s your usual demurrage structure?
                    Cheers,
                    Reza
                """,

                """
                    Good day Mr. Ingram,
                    We’d like to schedule 4,800MT of isopropanol from Houston to Cartagena. Could you give us a rate estimate, and do you handle cargo insurance under DAP terms?
                    Regards,
                    Lucia
                """,

                """
                    Jonas,
                    Could you provide a freight quote for 6,400MT of bitumen from Bahrain to Mombasa in late July? Also, do you see any risk of port congestion on your end?
                    Many thanks,
                    Noah
                """,

                """
                    Hi Kim,
                    We have 10.2kMT of gasoline ready in Sikka bound for Singapore. Could you share your best offer, and is there any discount if laytime is under 36 hours?
                    Best,
                    Anil
                """,

                """
                    Hello Charlene,
                    Any chance you can fix a vessel this Friday for 3,300MT of lubricants from Genoa to Haifa? And how soon would you need the charter party signed?
                    Kind regards,
                    Giulia
                """,

                """
                    Sven,
                    We’re interested in 7,500MT of chemicals from Hamburg to Gdansk. Could you send your rate ideas, and is your vessel flexible on partial loading windows?
                    Sincerely,
                    Jonas
                """,

                """
                    Hi Mr. Novak,
                    We plan to ship 8,000MT of diesel from Gdynia to Hull. Could you confirm available laycan slots next week, and do you anticipate any additional bunker surcharge?
                    Regards,
                    Irina
                """,

                """
                    Hello Ms. Olsen,
                    Could you handle a 2,600MT cargo of adhesives from Antwerp to Liverpool before month’s end? Also, can you provide your typical demurrage and dispatch rates?
                    Thanks,
                    Patrick
                """,

                """
                    Good morning Mr. Patel,
                    We’d like a quote for 5,500MT of soybean oil from Paranaguá to Ravenna. Do you foresee any delays with customs clearance, and what’s your lumpsum idea?
                    Warm regards,
                    Camila
                """,

                """
                    Ms. Quintero,
                    We have 4,000MT of refined sugar destined for Port Said from Valencia. Could you confirm your earliest vessel position, and is a berth guaranteed on arrival?
                    Best,
                    Enrique
                """,

                """
                    Hey Mr. Russell,
                    We’re looking to move 3,200MT of steel coils from Dalian to Kaohsiung. Could you advise a ballpark freight rate, and do you typically quote under an FIO basis?
                    Cheers,
                    Roger
                """,

                """
                    Hello Tetsu,
                    We’re arranging 9,300MT of vegetable oil from Busan to Port Klang. Could you let me know if your vessel is open next week, and how flexible you are with laytime extensions?
                    Sincerely,
                    Min
                """,

                """
                    Hi Ms. Thompson,
                    We want to move 6,600MT of polymers from Jubail to Barcelona. Can you send a freight estimate, and will you charge any additional fees if we miss the narrow laycan?
                    Thanks,
                    Rafael
                """,

                """
                    Dear Mr. Uddin,
                    Are you available to take 5,900MT of fertilizer from Aqaba to Jeddah this weekend? Also, could you clarify if pilotage and towage fees are included in your quote?
                    Thank you,
                    Ziad
                """,

                """
                    Mark,
                    We need to ship 7,100MT of kerosene from Rotterdam to Las Palmas. Do you have any open slots in early September, and can you confirm if your vessel fits draft restrictions?
                    Best,
                    Eva
                """,

                """
                    Hi Eva,
                    Could you handle 4,500MT of industrial solvents from Hamriyah to Mumbai on short notice? And what would be your demurrage rate after 72 hours of laytime?
                    Regards,
                    Amit
                """,

                """
                    We have a 9,700MT cargo of naphtha loading at Sikka for Piraeus. Do you have capacity within the next 10 days, and can you match the freight rate we had last month?
                    Cheers,
                    Helena
                """,

                """
                    Hello Ms. Yoon,
                    We’re finalizing 8,800MT of crude palm oil from Belawan to Yangon. Could you provide a quick estimate for freight, and do you foresee any port draft limitations?
                    Kindly,
                    Sofia
                """,

                """
                    Hello Ms. Adler,
                    We have 3,500MT of polypropylene ready from Al Jubail to Trieste next month. Could you confirm if partial cargo is acceptable and provide your lumpsum rate?
                    Sincerely,
                    Warren
                """,

                """
                    Morning Mr. Bailey,
                    I’m arranging 4,300MT of wheat from Odessa to Barcelona. Could you advise if you can fit this in early August, and what's your standard demurrage clause?
                    Best regards,
                    Vicente
                """,

                """
                    Hey Davin,
                    We’re looking at shipping 6,2k MT of sodium hydroxide from Busan to Hong Kong. Do you happen to have any boat open next week, and how flexible is your laycan?
                    Thanks,
                    Luca
                """,

                """
                    Hi Lucia,
                    We plan to move 2,900MT of corn from Veracruz to Tampa. Could you lmk if wknd loadings incur an extra fee, and do you anticipate any dock delays?
                    Warmly,
                    Helena
                """,

                """
                    Hello Ms. Evans,
                    We have an urgent need for 5,000MT of LNG from Ras Laffan to Jebel Ali. Could you confirm vessel availability by the 10th, and do you allow part cargo?
                    Regards,
                    Miguel
                """,

                """
                    Matthieu,
                    We’d like to fix 7,100MT of bitumen from Jubail, to Salalah next month.. Could you share your best freight idea and vessel name?
                    Thanks,
                    Caroline
                """,

                """
                    Hey Ms. Garcia,
                    Could you handle 9,400MT of soybean oil from Paranaguá to Dakar? And do you foresee any additional bunker surcharges in the current market?
                    Best wishes,
                    Randall
                """,

                """
                    Hello Mr. Hampton,
                    We’re exploring rates for 4,500MT of adhesives from Hamburg to Aarhus. Would it be possible to combine cargo with any existing fixture, and what’s your demurrage rate?
                    Thanks,
                    Ivy
                """,

                """
                    Hello Ms. Ingram,
                    We want   to ship 2,700MT of vegetable oil from Santos to Freetown. Could you confirm your earliest laycan slot    and clarify if you quote under CIF?
                    Kind regards,
                    Richard
                """,

                """
                    Hi James,
                    Looking to move 5,8kmt of palm kernel oil from Port Klang to Male. Could you give me an indication of freight, and is laytime negotiable beyond 48 hours?
                    Cheers,
                    Mona
                """,

                """
                    Good day Ms. Kelly,
                    I’m inquiring about 3,400MT of sulphuric acid from Aqaba to Khor Fakkan. Could you handle a late-July laycan, and are you open to a short load window?
                    Best,
                    Rami
                """,

                """
                    Hey Mr. Larsen,
                    Do you have a suitable vessel for 6,300MT of pig iron from Gijon to Antwerp next week? ?
                    Thank you,
                    Olivia
                """,

                """
                    Hello Ms. Moore,
                    We’re looking at 8,500MT of sugar from Santos to Libreville. Could you confirm your demurrage rate, and do you anticipate any weekend berth restrictions?
                    Warm regards,
                    Jerome
                """,

                """
                    Hi Mr. Nelson,
                    We have 7,800MT of steel plates from Kaohsiung to Jakarta. Could you please provide a lumpsum freight quote and confirm if partial discharge is allowed?
                    Sincerely,
                    Alma
                """,

                """
                    Morning Ms. Orlov,
                    Could you fix 3,100MT of coffee beans from Ho Chi Minh to Busan for mid-month? Also, do you typically charge extra for reefer plugs?
                    Respectfully,
                    Kurt
                """,

                """
                    Hey Mr. Park,
                    We're scheduling 5,400MT of barley from Constanța to Beirut. Do you have space this weekend, and can we finalize the rate within two days?
                    Best,
                    Yasmine
                """,

                """
                    Hello Quinn,
                    We’d need to load 2,200MT of polymers from Fujairah to Dar es Salaam. Could you advise if you can handle a quick turnaround, and do you allow direct voyage?
                    Thanks,
                    Marvin
                """,

                """
                    Hi Mr. Roberts,
                    Plase confirm if your vessel can take 9,000MT of PET resin from Mundra to Port Sudan??? Also, do you forsee any exra pilotage charges?
                    Warmly,
                    Beth
                """,

                """
                    Hello Ms. Saunders,
                    We're looking to ship 4.t kmt of crude from Belawan to Penang. Any possibility to load by the 12th, and what is the loading rate per hour?
                    Cheers,
                    Lina
                """,

                """
                    Dear Mr. Taylor,
                    Could you provide a freight estimate for 3,900MT of ethanol from Maceió to Houston in late August? Also, is laytime counted on arrival or upon NOR?
                    Many thanks,
                    Felix
                """,

                """
                    Hi,
                    We need 6,600MT capacity for coal from Richards Bay to Maputo next week. Do you have an open slot?
                    Regards,
                    Adrian
                """,

                """
                    Hey Vincent,
                    Is your vessel available to lift 5,100MT of diesel from Abu Dhabi to Djibouti early September? Also, could we arrange partial payment upfront?
                    Sincerely,
                    Carol
                """,

                """
                    Good afternoon Ms. Watson,
                    Have you heared about the oil spillage of MS High Hope last week in the gulf of mexico? Any deatails you can share?
                    Regards,
                    Rory
                """,

                """
                    Hello Mr. Xander,
                    Interested in shipping 7,200MT of biodiesel from Singapore to Yangon. Do you handle bunkers on owners’ account, and can you accommodate a 3-day laycan extension?
                    Thank you,
                    Elise
                """,

                """
                    Hi Ms. Young,
                    We plan to move 4,000MT of adhesives from Le Havre to Casablanca. Could you provide a lumpsum freight rate, and is your vessel flexible on draft requirements?
                    Warmly,
                    Eman
                """,

                """
                    Good day Mr.Zhou,
                    We    have about 8,100MT of rapeseed oil from Vancouver to Busan. Coud you confirm rate? Also is weekend loading an option?
                    Thank you,
                    Alicia
                """,

                """
                    Hi Ms. Abram,
                    Could you handle 3,300MT of cocoa beans from Abidjan to Tarragona next month, and what's your typical dispatch rate?
                    Best,
                    Franco
                """,

                """
                    Dear Mr. Bravo,
                    We need a vessel for 6,900MT of liquefied petroleum gas from Bahrain to Mina Al Ahmadi. Could you advise on your next open laycan, and do you accept lumpsum deals?
                    Regards,
                    Tiana
                """,

                """
                    Hello Ms. Chen,
                    The market for lng carriers seems to tighten. Where do you foresee the rates within the next 2-3 months?
                    Many thanks,
                    Andre
                """,

                """
                    Hi Mr. Davis,
                    For 2,700MT of ammonium sulfate from Damietta to Koper, do you have capacity next week, and can we finalize a CP with minimal laycan buffer?
                    Cheers,
                    Maria
                """,

                """
                    Hey Eddy,
                    To celebrate year's end, we'd like to invite you to our christmas party on 20th Dec. 2024 at our offices in New York. Are you able to join?
                    Looking forward,
                    Max
                """,

                """
                    Hello Mr. Franco,
                    Can your vsl lift 8,400MT of clinker from Sohar to Mombasa? Also, do you apply a bunker escalation clause or keep a fixed rate?
                    Warm regards,
                    Paula
                """,

                """
                    Rob,
                    Heared rumors that you are in talks to merge with Broker inc, is this true?
                    Thanks,
                    Jorge
                """,

                """
                    Hi Kathy,
                    We heared that our competitors are in the market to ship 10k crude from Rio to Algeciras. Do you know their fixing level?
                    Regards,
                    Tabitha
                """,

                """
                    Hello Ms. Irons,
                    We want to fix 4,600MT of lubricants from Port Sudan to Jebel Ali. Could you confirm your lumpsum offer, and do you see any transit congestion?
                    Kindly,
                    Patrick
                """,

                """
                    Dear Mr Johnson,
                    We have 9,100MT of sttyrene from Singapore to Manila??? Could yoy give me a ballpark rate asap?? Also, is partial load feasible???
                    Best,
                    Amin
                """,

                """
                    Hey Jonny,
                    Looking to move 6k MT of bulk corn starch from Bangkok to Haiphong. Could you provide a preliminary freight indication for CFR?
                    Sincerely,
                    Zara
                """,

                """
                    Good day Mr. Lewis,
                    Please note that Mr. Green will leave our brokering firm at the end of the month. I will take over our contract and would be happy to meet. Any availability early next week?
                    Many thanks in advance and best regards,
                    Daniel James
                """,

                """
                    Hello Ms. Morrow,
                    We’re scheduling a 3,200MT cargo of phosphates from Casablanca to Lisbon. Can you quote a vessel with laycan after the 20th?
                    Regards,
                    Ishaan
                """,

                # 40
                """
                    Hi Neal,
                    We need a tanker for 4,400MT of white spirit from Riga to Dublin. Could you confirm your earliest berth availability, and do you typically quote under FOB?
                    Best,
                    Mei
                """,

                """
                    Dear Ms. Ocampo,
                    We require shipping diesel (8k mt) from Fujairah to Berbera. Can you get us your best quote basis CFR?
                    Thx,
                    Louis
                """,

                """
                    Hello Mr. Perez,
                    We are still awaiting payment for your shipment of coal from last month. Can you provide us with a value date?
                    Warmly,
                    Zoe Cortez
                """,

                """
                    Hey Kenny,
                    We have 4,900MT of scrap metal from Gdansk to Rotterdam. Do you happen to have a close opening next week?
                    Kind regards,
                    Tomas
                """,

                # 44
                """
                    Good evening Mr. Stone,
                    Seeking 6,800MT capacity for wood pulp from Vancouver to Yokohama. Could you advise your best freight estimate, and do you allow weekend loading?
                    Cheers,
                    Natalie
                """,

                """
                    Hi Ms. Taylor,
                    For the shipment of coal under MS Irene, we need to provision owners vessel hoses. Can you please send us the LOI and advise if any upcharge on the freight?
                    Thank you,
                    Rohan
                """,

                """
                    Hello Adi,
                    Exploring 20kt shipment of sul acid from Antwerp to Houston. Can you provide some vessel options basis CFR?
                    Best regards,
                    Karina
                """,

                """
                    Hi Ms. Vargas,
                    Can you please provide us with an updated ETA for MS Mangrovia?
                    Sincerely,
                    Anders
                """,

                """
                    Hey Mr. Wallace,
                    We’re sourcing a 7,400MT cargo of crude palm oil from Port Klang to Surabaya. Do you have a vessel open in the next two weeks, and what’s your demurrage policy?
                    Many thanks,
                    Clara
                """,

                """
                    Hello Ms. Xue,
                    Interested in 3,700MT of paraffin wax from Daesan to Kaohsiung. Could you handle it under FOB terms, and do you foresee any weather-related delays?
                    Regards,
                    Tommy
                """,

                """
                    Hi Mr. Yang,
                    Could your vessel load 6,500MT of naphtha from Yanbu to Karachi next week, and do you typically finalize freight within 48 hours of quoting?
                    Thanks,
                    Beatrice
                """
            ]