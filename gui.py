import os
import sys
import json
import ast
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk
import re


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        def exit_gui():
            parameter_setting = {"INPUT_PATH": folder_path.get(),
                                 "ERODE_THICKNESS": erode_thickness.get(),
                                 "RING_THICKNESS": ring_thickness.get(),
                                 "RDL_AVER": rdl_aver.get(),
                                 "RDL_AVER_START": rdl_aver_start.get() if rdl_aver.get() else 0,
                                 "RDL_AVER_END": rdl_aver_end.get() if rdl_aver.get() else 0,
                                 "BK_SUB": background_sub.get(),
                                 "SHOW_OVERLAP": show_overlap.get(),
                                 "SHOW_PLOTS": show_plots.get(),
                                 "CHANNELS": ch_info,
                                 "B_CHANNEL": ch_base.get()
                                 }
            print(str(parameter_setting))

            if rdl_aver_end.get() > ring_thickness.get():
                err_msg_lb.configure(text="Wrong setting: end layer exceed thickness!")

            elif rdl_aver_start.get() > ring_thickness.get():
                err_msg_lb.configure(text="Wrong setting: tarting layer exceed thickness!")

            elif rdl_aver_start.get() > rdl_aver_end.get():
                err_msg_lb.configure(text="Wrong setting: starting layer smaller than end layer!")

            else:
                err_msg_lb.configure(text="valid setting, proceed to analysis...")
                parameters = str(parameter_setting).replace(" ", "")
                os.system("python main.py " + json.dumps(parameters))
                self.destroy()

        def get_input_path():
            INPUT_PATH = r'{}'.format(askdirectory(initialdir=r'./input', mustexist=True))
            folder_path.set(INPUT_PATH)
            print(INPUT_PATH)

        def erode_slider_changed(event):
            erode_slide_num_lb.configure(text=get_current_value(erode_thickness))

        def ring_slider_changed(event):
            # ring_slider.get()
            # rdl_aver_start_max.set(ring_thickness.get())
            # print(rdl_aver_start_max.get())
            # print(int(event))
            ring_slide_num_lb.configure(text=get_current_value(ring_thickness))
            if rdl_aver.get():
                rdl_aver_start_slider.configure(to=ring_thickness.get())
                rdl_aver_start_slider.update_idletasks()
                rdl_aver_end.set(ring_thickness.get())
                rdl_aver_end_slider_num_lb.configure(text=get_current_value(rdl_aver_end))
                rdl_aver_end_slider.configure(to=ring_thickness.get())
                rdl_aver_end_slider.update_idletasks()

        def rdl_aver_start_slider_changed(event):
            # ring_slider.get()
            rdl_aver_end_slider.configure(from_=rdl_aver_start.get())
            rdl_aver_end_slider.update_idletasks()
            rdl_aver_start_slider_num_lb.configure(text=get_current_value(rdl_aver_start))

        def rdl_aver_end_slider_changed(event):
            # ring_slider.get()
            rdl_aver_end_slider_num_lb.configure(text=get_current_value(rdl_aver_end))
            rdl_aver_start_slider.configure(to=rdl_aver_end.get())
            rdl_aver_start_slider.update_idletasks()

        def get_current_value(part):
            return part.get()

        def toggle_rdl_aver():
            print(rdl_aver.get())
            if rdl_aver.get():
                rdl_aver_start_slider.configure(state=tk.ACTIVE)
                rdl_aver_end_slider.configure(state=tk.ACTIVE)
            else:
                rdl_aver_start_slider.configure(state=tk.DISABLED)
                rdl_aver_end_slider.configure(state=tk.DISABLED)

                rdl_aver_start.set(1)
                rdl_aver_end.set(1)
                rdl_aver_start_slider_num_lb.configure(text=rdl_aver_start.get())
                rdl_aver_end_slider_num_lb.configure(text=rdl_aver_end.get())

        def pick_ch_num(event):
            if ch_num.get() in ch_info.keys():
                ch_name.set(ch_info[ch_num.get()])
            else:
                ch_name.set('')

        def add_channel():
            if re.fullmatch('[a-zA-Z0-9]+', ch_name.get()) is None:
                ch_name.set('only digits/letters')
            else:
                ch_info[ch_num.get()] = ch_name.get()
                if ch_num.get() == ch_num_list[-1]:
                    ch_num_list.append(ch_num.get()+1)
                    # ch_num.set(ch_num.get()+1)
                    ch_num_opt['menu'].delete(0, 'end')
                    ch_base_opt['menu'].delete(0, 'end')# remove full list
                    # ch_num_opt['menu'].add_command(command=pick_ch_num())
                    for opt in ch_num_list:
                        ch_num_opt['menu'].add_command(label=opt, command=tk._setit(ch_num, opt, pick_ch_num))
                        ch_base_opt['menu'].add_command(label=opt, command=tk._setit(ch_base, opt))
                    ch_num.set(ch_num_list[-1])
                    ch_name.set('')

        # def show_message(error='', color='black'):
        #     label_error['text'] = error
        #     email_entry['foreground'] = color

        def validate(value):
            """
            Validat the email entry
            :param value:
            :return:
            """
            if re.fullmatch('[a-zA-Z0-9]+', value) is None:
                return False

            # show_message()
            return True

        def on_invalid():
            """
            Show the error message if the data is not valid
            :return:
            """
            ch_name.set('Only digits/letters')
            # part.show_message('Please enter a valid email', 'red')



        self.geometry('450x480')
        self.resizable(0, 0)
        self.title('COSIMA')

        # UI options
        paddings = {'padx': 5, 'pady': 5}
        entry_font = {'font': ('Helvetica', 11)}

        # configure the grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        ch_num_list = [0]
        ch_num = tk.IntVar(self, value=ch_num_list[0])
        # ch_num.set(ch_num_list[0])

        # ch_num_list = list
        ch_nums = list
        ch_names = list
        ch_info = {
            0: '',
        }

        folder_path = tk.StringVar(self, value=r'./input')
        erode_thickness = tk.IntVar(self, value=0)
        ring_thickness = tk.IntVar(self, value=1)
        rdl_aver_start = tk.IntVar(self, value=1)
        rdl_aver_start_max = tk.IntVar(self, value=20)
        rdl_aver_end = tk.IntVar(self, value=1)
        rdl_aver_end_max = tk.IntVar(self, value=1)
        rdl_aver_end_min = tk.IntVar(self, value=20)
        background_sub = tk.BooleanVar(self, value=True)
        show_overlap = tk.BooleanVar(self, value=True)
        show_plots = tk.BooleanVar(self, value=True)
        rdl_aver = tk.BooleanVar(self, value=True)
        ch_base = tk.IntVar(self, value=0)
        ch_name = tk.StringVar(self)

        # heading
        heading = ttk.Label(self, text='Parameters Setup', style='Heading.TLabel')
        heading.grid(column=0, row=0, columnspan=3, pady=5, sticky=tk.N)

        # input path
        path_lb = ttk.Label(self, text="Input path: ")
        path_lb.grid(column=0, row=1, sticky=tk.W, **paddings)

        path_entry = ttk.Entry(self, textvariable=folder_path, state="disabled", **entry_font)
        path_entry.grid(column=1, row=1, sticky=tk.EW, **paddings)

        path_entry = ttk.Button(self, text="Select", command=get_input_path)
        path_entry.grid(column=2, row=1, sticky=tk.E, **paddings)

        # erode
        erode_lb = ttk.Label(self, text="Erode:")
        erode_lb.grid(column=0, row=2, sticky=tk.W, **paddings)

        erode_slider = ttk.Scale(
            self,
            from_=0,
            to=20,
            orient='horizontal',  # horizontal
            variable=erode_thickness,
            command=erode_slider_changed,
        )
        erode_slider.grid(column=1, row=2, sticky=tk.EW, **paddings)

        erode_slide_num_lb = ttk.Label(
            self,
            text=get_current_value(erode_thickness)
        )
        erode_slide_num_lb.grid(column=2, row=2, sticky=tk.W, **paddings)

        # thickness
        thickness_lb = ttk.Label(self, text="Thickness:")
        thickness_lb.grid(column=0, row=3, sticky=tk.W, **paddings)

        ring_slider = ttk.Scale(
            self,
            from_=1,
            to=20,
            orient='horizontal',  # horizontal
            variable=ring_thickness,
            command=ring_slider_changed,
        )
        ring_slider.grid(column=1, row=3, sticky=tk.EW, **paddings)

        ring_slide_num_lb = ttk.Label(
            self,
            text=get_current_value(ring_thickness)
        )
        ring_slide_num_lb.grid(column=2, row=3, sticky=tk.W, **paddings)

        # radio average setting
        rdl_aver_lb = ttk.Label(self, text="Radial Average:")
        rdl_aver_lb.grid(column=0, row=4, columnspan=2, sticky=tk.W, **paddings)

        rdl_aver_check = ttk.Checkbutton(
            self,
            variable=rdl_aver,
            command=toggle_rdl_aver
        )
        rdl_aver_check.grid(column=1, row=4, sticky=tk.E, **paddings)

        rdl_aver_start_lb = ttk.Label(self, text="Starting Layer:")
        rdl_aver_start_lb.grid(column=0, row=5, sticky=tk.W, **paddings)

        rdl_aver_start_slider = ttk.Scale(
            self,
            from_=1,
            to=rdl_aver_end.get(),
            orient='horizontal',  # horizontal
            variable=rdl_aver_start,
            command=rdl_aver_start_slider_changed,
        )
        rdl_aver_start_slider.grid(column=1, row=5, sticky=tk.EW, **paddings)

        rdl_aver_start_slider_num_lb = ttk.Label(
            self,
            text=get_current_value(rdl_aver_start)
        )
        rdl_aver_start_slider_num_lb.grid(column=2, row=5, sticky=tk.W, **paddings)

        rdl_aver_end_lb = ttk.Label(self, text="End Layer:")
        rdl_aver_end_lb.grid(column=0, row=6, sticky=tk.W, **paddings)

        rdl_aver_end_slider = ttk.Scale(
            self,
            from_=1,
            to=ring_thickness.get(),
            orient='horizontal',  # horizontal
            variable=rdl_aver_end,
            command=rdl_aver_end_slider_changed,
        )
        rdl_aver_end_slider.grid(column=1, row=6, sticky=tk.EW, **paddings)

        rdl_aver_end_slider_num_lb = ttk.Label(
            self,
            text=get_current_value(rdl_aver_end)
        )
        rdl_aver_end_slider_num_lb.grid(column=2, row=6, sticky=tk.W, **paddings)

        # background sub
        bk_sub_lb = ttk.Label(self, text="Background subtraction:")
        bk_sub_lb.grid(column=0, row=7, columnspan=2, sticky=tk.W, **paddings)

        bk_sub_check = ttk.Checkbutton(
            self,
            variable=background_sub,

        )
        bk_sub_check.grid(column=1, row=7, sticky=tk.E, **paddings)

        # show overlap
        show_overlap_lb = ttk.Label(self, text="Show Overlapped pixel on plots:")
        show_overlap_lb.grid(column=0, row=8, columnspan=2, sticky=tk.W, **paddings)

        show_overlap_check = ttk.Checkbutton(
            self,
            variable=show_overlap,
        )
        show_overlap_check.grid(column=1, row=8, sticky=tk.E, **paddings)

        # show plots
        show_plots_lb = ttk.Label(self, text="Show plots:")
        show_plots_lb.grid(column=0, row=79, columnspan=2, sticky=tk.W, **paddings)

        show_plots_check = ttk.Checkbutton(
            self,
            variable=show_plots,
        )
        show_plots_check.grid(column=1, row=79, sticky=tk.E, **paddings)

        # Channel number label
        ch_num_lb = ttk.Label(self, text="Channel:")
        ch_num_lb.grid(column=0, row=80, columnspan=1, sticky=tk.W, **paddings)

        ch_num_opt = ttk.OptionMenu(
            self,
            ch_num,
            # ch_num_list[0],
            *ch_num_list,
            command=pick_ch_num

        )
        ch_num_opt.grid(column=0, row=80, sticky=tk.E, **paddings)

        ch_name_entry = ttk.Entry(self, textvariable=ch_name,)
        vcmd = (ch_name_entry.register(validate), '%P')
        ivcmd = (ch_name_entry.register(on_invalid),)
        ch_name_entry.config(validate='focusout', validatecommand=vcmd, invalidcommand=ivcmd)
        ch_name_entry.grid(column=1, row=80, columnspan=1, sticky=tk.EW, **paddings)
        add_ch_btn = ttk.Button(self, text="Add", command=add_channel)
        add_ch_btn.grid(column=2, row=80, sticky=tk.E, **paddings)

        # base select
        ch_base_lb = ttk.Label(self, text="Base CH:")
        ch_base_lb.grid(column=0, row=81, columnspan=1, sticky=tk.W, **paddings)

        ch_base_opt = ttk.OptionMenu(
            self,
            ch_base,
            # ch_num_list[0],
            *ch_num_list,


        )
        ch_base_opt.grid(column=0, row=81, sticky=tk.E, **paddings)

        # error message
        err_msg_lb = ttk.Label(self, text="")
        err_msg_lb.grid(column=0, row=89, columnspan=3, sticky=tk.W, **paddings)

        # run button
        login_button = ttk.Button(self, text="Run", command=exit_gui)
        login_button.grid(column=2, row=99, sticky=tk.E, **paddings)

        # configure style
        self.style = ttk.Style(self)
        self.style.configure('TLabel', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 11))

        # heading style
        self.style.configure('Heading.TLabel', font=('Helvetica', 12))


app = App()
app.mainloop()

